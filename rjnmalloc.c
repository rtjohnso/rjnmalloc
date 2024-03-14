#include "rjnmalloc.h"
#include <assert.h>
#include <immintrin.h>
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#define debug_assert(x) assert(x)
//#define debug_assert(x)

struct rjn_allocator {
  size_t region_size;
  size_t allocation_unit_size;
  uint64_t num_allocation_units;

  // Offset, in bytes, from start of this header, to the first allocation_unit.
  // The allocation units are located immediately after this header,
  // except for possible padding for alignment.
  uint64_t allocation_units_offset;

  // Offsets, in bytes, from start of this header, to the metadata vector and
  // size_class array, respectively.  The metadata vector and
  // size-class array are located at the _end_ of the region, which
  // should make it easier to grow regions in the future.
  uint64_t metadata_offset;
  uint64_t size_classes_offset;
};

typedef struct rjn_node {
  uint64_t prev; // offset, in bytes, from start of rjn_allocator
  uint64_t next; // offset, in bytes, from start of rjn_allocator
  uint64_t size; // size of this chunk in allocation units
                 // the size field is also used as a spinlock for the size class
} rjn_node;

typedef struct size_class {
  rjn_node head;
  uint64_t num_chunks;
  uint64_t num_units;
} size_class;

#define MIN_ALLOCATION_UNIT_SIZE (sizeof(rjn_node))

uint64_t rjn_offset(rjn_allocator *rjn, void *ptr) {
  return (uint8_t *)ptr - (uint8_t *)rjn;
}

static void *rjn_pointer(rjn_allocator *rjn, uint64_t offset) {
  return (void *)((uint8_t *)rjn + offset);
}

static uint8_t *rjn_allocation_units(rjn_allocator *rjn) {
  return (uint8_t *)rjn_pointer(rjn, rjn->allocation_units_offset);
}

static rjn_node *rjn_allocation_unit(rjn_allocator *rjn, uint64_t i) {
  assert(i < rjn->num_allocation_units);
  uint8_t *p = rjn_allocation_units(rjn);
  return (rjn_node *)(p + i * rjn->allocation_unit_size);
}

static uint8_t *rjn_metadata_vector(rjn_allocator *rjn) {
  return (uint8_t *)rjn_pointer(rjn, rjn->metadata_offset);
}

// Size class i contains chunks of size in the range [2^i, 2^(i+1)) allocation
// units.
static uint64_t rjn_num_size_classes(rjn_allocator *rjn) {
  return 64 - _lzcnt_u64(rjn->num_allocation_units);
}

static size_class *rjn_size_classes(rjn_allocator *rjn) {
  return (size_class *)rjn_pointer(rjn, rjn->size_classes_offset);
}

/* Return the size class where a free chunk of size 'units' allocation units
   should be stored. */
static unsigned int rjn_find_size_class_index(rjn_allocator *rjn,
                                              size_t units) {
  uint64_t num_sclasses = rjn_num_size_classes(rjn);

  unsigned int fl = 63 - _lzcnt_u64(units);
  if (num_sclasses <= fl) {
    return num_sclasses;
  } else {
    return fl;
  }
}

static size_class *rjn_find_size_class(rjn_allocator *rjn, size_t units) {
  unsigned int scidx = rjn_find_size_class_index(rjn, units);
  if (scidx < rjn_num_size_classes(rjn)) {
    return &rjn_size_classes(rjn)[scidx];
  } else {
    return &rjn_size_classes(rjn)[scidx - 1];
  }
}

static void rjn_lock_size_class(size_class *sc) {
  while (__sync_lock_test_and_set(&sc->head.size, 1)) {
    while (sc->head.size) {
      _mm_pause();
    }
  }
}

static void rjn_unlock_size_class(size_class *sc) { sc->head.size = 0; }

static void rjn_prepend(rjn_allocator *rjn, size_class *sclass,
                        rjn_node *node) {
  debug_assert(node->next == 0);
  debug_assert(node->prev == 0);
  node->next = sclass->head.next;
  node->prev = rjn_offset(rjn, sclass);
  if (sclass->head.next) {
    rjn_node *next = rjn_pointer(rjn, sclass->head.next);
    debug_assert(next->prev == rjn_offset(rjn, sclass));
    next->prev = rjn_offset(rjn, node);
  }
  sclass->head.next = rjn_offset(rjn, node);
  sclass->num_chunks++;
  sclass->num_units += node->size;
}

static void rjn_remove(rjn_allocator *rjn, size_class *sclass, rjn_node *node) {
  debug_assert(node->prev != 0);
  rjn_node *prev = rjn_pointer(rjn, node->prev);
  debug_assert(prev->next == rjn_offset(rjn, node));
  prev->next = node->next;
  if (node->next) {
    rjn_node *next = rjn_pointer(rjn, node->next);
    debug_assert(next->prev == rjn_offset(rjn, node));
    next->prev = node->prev;
  }
  node->next = node->prev = 0;
  sclass->num_chunks--;
  sclass->num_units -= node->size;
}

#define RJN_META_FREE (0)
#define RJN_META_UNARY (1)
#define RJN_META_BINARY (2)
#define RJN_META_CONTINUATION (3)
#define RJN_MIN_BINARY_UNITS (10)

static int rjn_metadata_cas(rjn_allocator *rjn, uint64_t unit, uint8_t old,
                            uint8_t new) {
  uint8_t *metadata = rjn_metadata_vector(rjn);
  return __sync_bool_compare_and_swap(&metadata[unit], old, new);
}

static void rjn_metadata_set(rjn_allocator *rjn, uint64_t unit, uint8_t value) {
  uint8_t *metadata = rjn_metadata_vector(rjn);
  metadata[unit] = value;
}

static void rjn_metadata_set_size(rjn_allocator *rjn, uint64_t first_unit,
                                  uint64_t units) {

  uint8_t *metadata = rjn_metadata_vector(rjn);
  if (units < RJN_MIN_BINARY_UNITS) {
    for (uint64_t i = 1; i < units - 1; i++) {
      rjn_metadata_set(rjn, first_unit + i, RJN_META_CONTINUATION);
    }
  } else {
    rjn_metadata_set(rjn, first_unit, RJN_META_BINARY);
    memcpy(&metadata[first_unit + 1], &units, sizeof(uint64_t));
  }
}

static uint64_t rjn_metadata_get_size(rjn_allocator *rjn, uint64_t first_unit) {
  uint8_t *metadata = rjn_metadata_vector(rjn);
  uint64_t size;
  if (metadata[first_unit] == RJN_META_UNARY) {
    size = 1;
    while (metadata[first_unit + size] == RJN_META_CONTINUATION) {
      size++;
    }
  } else {
    memcpy(&size, &metadata[first_unit + 1], sizeof(uint64_t));
  }
  return size;
}

static uint64_t rjn_grab_predecessor_if_free(rjn_allocator *rjn,
                                             uint64_t my_start) {
  if (my_start == 0) {
    return 0;
  }
  uint64_t pred_last = my_start - 1;
  if (!rjn_metadata_cas(rjn, pred_last, RJN_META_FREE, RJN_META_UNARY)) {
    return my_start;
  }
  rjn_node *pred_node = rjn_allocation_unit(rjn, pred_last);
  assert(pred_node->size <= my_start);
  uint64_t pred_start = my_start - pred_node->size;
  if (pred_node->size == 0 ||
      (pred_start < pred_last &&
       !rjn_metadata_cas(rjn, pred_start, RJN_META_FREE, RJN_META_UNARY))) {
    rjn_metadata_set(rjn, pred_last, RJN_META_FREE);
    return my_start;
  }
  return pred_start;
}

static uint64_t rjn_grab_successor_if_free(rjn_allocator *rjn,
                                           uint64_t my_start,
                                           uint64_t my_units) {
  uint64_t succ_start = my_start + my_units;
  if (rjn->num_allocation_units <= succ_start) {
    return 0;
  }
  if (!rjn_metadata_cas(rjn, succ_start, RJN_META_FREE, RJN_META_UNARY)) {
    return 0;
  }
  rjn_node *succ_node = rjn_allocation_unit(rjn, succ_start);
  while (succ_node->size == 0) {
    _mm_pause();
  }
  uint64_t succ_last = succ_start + succ_node->size - 1;
  if (succ_start < succ_last) {
    while (!rjn_metadata_cas(rjn, succ_last, RJN_META_FREE, RJN_META_UNARY)) {
      _mm_pause();
    }
  }
  return succ_start;
}

/* The goal of this code is to ensure that we always merge adjacent free chunks,
   which is tricky since two adjacent chunks may be getting freed concurrently.
   In that case, we have to ensure that one of the threads notices the other
   chunk is free (or becoming free), so that it will initiate a merge.
   We accomplish this by partially freeing our chunk and then checking whether
   our predecessor is free.  If it is, we merge with it.  If not, then the
   thread freeing our predecessor will be able to see that our chunk is
   (partially or completely) free and will hence perform the required merged.

   The way we "partially free" a chunk is as follows:
   - For multi-unit chunks, we mark the first unit as free.  This allows a
   thread freeing our predecessor chunk to see that we are in the process of
   being freed, but it will still have to wait for us to finish freeing our
   chunk before it can merge with us, because it will have to wait for us to
   mark the last unit as free.
   - For single-unit chunks, we can't play the above trick.  So we set the size
   field in the unit to 0, which is a special value that indicates that the
   chunk is in the process of being freed.  This allows a thread freeing our
   predecessor to see that we are in the process of being freed, and it will be
   able to merge with us when we indicate that we are done (by setting the size
   to 1).
 */
static void rjn_free_chunk(rjn_allocator *rjn, uint64_t start_unit,
                           uint64_t units) {
  rjn_node *start_node;
restart:
  start_node = rjn_allocation_unit(rjn, start_unit);
  rjn_node *last_node = rjn_allocation_unit(rjn, start_unit + units - 1);

  if (1 < units) {
    start_node->size = units;
    last_node->size = units;
  } else {
    start_node->size = 0;
  }

  rjn_metadata_set(rjn, start_unit, RJN_META_FREE);

  uint64_t new_start = rjn_grab_predecessor_if_free(rjn, start_unit);
  if (new_start < start_unit) {
    rjn_node *pred_start_node = rjn_allocation_unit(rjn, new_start);
    size_class *pred_sc = rjn_find_size_class(rjn, pred_start_node->size);
    rjn_lock_size_class(pred_sc);
    rjn_remove(rjn, pred_sc, pred_start_node);
    rjn_unlock_size_class(pred_sc);
    start_unit = new_start;
    units += pred_start_node->size;
    goto restart;
  }

  size_class *sc = rjn_find_size_class(rjn, units);
  rjn_lock_size_class(sc);

  if (1 < units) {
    rjn_metadata_set(rjn, start_unit + units - 1, RJN_META_FREE);
  } else {
    start_node->size = 1;
  }

  rjn_prepend(rjn, sc, start_node);
  rjn_unlock_size_class(sc);

  uint64_t successor_start = rjn_grab_successor_if_free(rjn, start_unit, units);
  if (successor_start) {
    rjn_node *succ_start_node = rjn_allocation_unit(rjn, successor_start);
    size_class *succ_sc = rjn_find_size_class(rjn, succ_start_node->size);
    rjn_lock_size_class(succ_sc);
    rjn_remove(rjn, succ_sc, succ_start_node);
    rjn_unlock_size_class(succ_sc);
    start_unit = successor_start;
    units = succ_start_node->size;
    goto restart;
  }
}

static void *rjn_alloc_from_size_class(rjn_allocator *rjn, unsigned int scidx,
                                       size_t alignment_bytes, size_t bytes) {
  size_class *sc = &rjn_size_classes(rjn)[scidx];
  if (!sc->num_chunks) {
    return NULL;
  }
  rjn_lock_size_class(sc);

  for (uint64_t curr_off = sc->head.next; curr_off;
       curr_off = ((rjn_node *)rjn_pointer(rjn, curr_off))->next) {

    uint64_t first_unit =
        (curr_off - rjn->allocation_units_offset) / rjn->allocation_unit_size;
    if (!rjn_metadata_cas(rjn, first_unit, RJN_META_FREE, RJN_META_UNARY)) {
      // If the CAS fails, that means that someone else is freeing the
      // previous chunk and merging their free chunk with the current one.
      continue;
    }
    rjn_node *node = rjn_pointer(rjn, curr_off);

    // Check if the current chunk is large enough to satisfy the request,
    // including alignment padding.
    uint64_t pad_bytes = 0;
    uint64_t pad_units = 0;
    if (1 < alignment_bytes) {
      pad_bytes = alignment_bytes - ((uint64_t)node % alignment_bytes);
      pad_units = pad_bytes / rjn->allocation_unit_size;
    }
    uint64_t required_bytes = pad_bytes + bytes;
    uint64_t required_units = (required_bytes + rjn->allocation_unit_size - 1) /
                              rjn->allocation_unit_size;
    uint64_t allocated_units = required_units - pad_units;

    if (node->size < required_units) {
      rjn_metadata_set(rjn, first_unit, RJN_META_FREE);
      continue;
    }

    // Finish allocating the chunk.
    if (1 < node->size) {
      while (!rjn_metadata_cas(rjn, first_unit + node->size - 1, RJN_META_FREE,
                               RJN_META_CONTINUATION)) {
        _mm_pause();
      }
    }
    rjn_remove(rjn, sc, node);
    rjn_unlock_size_class(sc);

    // Give back any alignment pad at the beginning
    if (pad_units) {
      if (1 < pad_units) {
        rjn_metadata_set(rjn, first_unit + pad_units - 1,
                         RJN_META_CONTINUATION);
      }
      rjn_metadata_set(rjn, first_unit + pad_units, RJN_META_UNARY);
      rjn_free_chunk(rjn, first_unit, pad_units);
    }

    // Give back any remaining space at the end
    if (required_units < node->size) {
      if (1 < allocated_units) {
        rjn_metadata_set(rjn, first_unit + pad_units + allocated_units - 1,
                         RJN_META_CONTINUATION);
      }
      rjn_metadata_set(rjn, first_unit + pad_units + allocated_units,
                       RJN_META_UNARY);
      rjn_free_chunk(rjn, first_unit + pad_units + allocated_units,
                     node->size - pad_units - allocated_units);
    }

    rjn_metadata_set_size(rjn, first_unit + pad_units, allocated_units);

    return (uint8_t *)node + pad_bytes;
  }
  rjn_unlock_size_class(sc);
  return NULL;
}

void *rjn_alloc(rjn_allocator *rjn, size_t alignment, size_t size) {
  // Emulate the behavior of malloc(0), which returns a different
  // pointer every time it is called.
  if (size == 0) {
    size = 1;
  }

  size_t units =
      (size + rjn->allocation_unit_size - 1) / rjn->allocation_unit_size;
  uint64_t num_sclasses = rjn_num_size_classes(rjn);

  // First search the larger size classes.  Any node in any of those size
  // classes should be sufficient to satisfy the request (except possibly due to
  // alignment constraints).
  for (unsigned int scidx = 1 + rjn_find_size_class_index(rjn, units);
       scidx < num_sclasses; scidx++) {
    void *ptr = rjn_alloc_from_size_class(rjn, scidx, alignment, size);
    if (ptr) {
      return ptr;
    }
  }
  // OK we're desperate.  Try groveling through the size class for this
  // particular size to see if there's anything there.
  return rjn_alloc_from_size_class(rjn, rjn_find_size_class_index(rjn, units),
                                   alignment, size);
}

void rjn_free(rjn_allocator *rjn, void *ptr) {
  uint64_t first_unit = (rjn_offset(rjn, ptr) - rjn->allocation_units_offset) /
                        rjn->allocation_unit_size;
  uint64_t size = rjn_metadata_get_size(rjn, first_unit);
  rjn_node *node = rjn_allocation_unit(rjn, first_unit);
  node->prev = node->next = 0;
  rjn_free_chunk(rjn, first_unit, size);
}

int rjn_init(rjn_allocator *rjn, size_t region_size,
             size_t allocation_unit_size) {
  if (region_size < sizeof(rjn_allocator)) {
    return -1;
  }
  if (allocation_unit_size < MIN_ALLOCATION_UNIT_SIZE) {
    return -1;
  }

  rjn->region_size = region_size;
  rjn->allocation_unit_size = allocation_unit_size;

  rjn->allocation_units_offset = sizeof(rjn_allocator);
  uint64_t alignment =
      ((uint64_t)rjn + rjn->allocation_units_offset) % allocation_unit_size;
  if (alignment) {
    rjn->allocation_units_offset += allocation_unit_size - alignment;
  }
  if (rjn->region_size < rjn->allocation_units_offset) {
    return -1;
  }

  uint64_t space = rjn->region_size - rjn->allocation_units_offset;
  // The +1 is for each allocation unit's metadata byte.
  rjn->num_allocation_units = space / (allocation_unit_size + 1);
  while (space < rjn->num_allocation_units * (allocation_unit_size + 1) +
                     sizeof(size_class) * rjn_num_size_classes(rjn)) {
    rjn->num_allocation_units--;
  }

  rjn->metadata_offset = rjn->region_size - rjn->num_allocation_units;
  rjn->size_classes_offset =
      rjn->metadata_offset - sizeof(size_class) * rjn_num_size_classes(rjn);

  memset(rjn_metadata_vector(rjn), RJN_META_FREE, rjn->num_allocation_units);

  size_class *scs = rjn_size_classes(rjn);
  uint64_t num_scs = rjn_num_size_classes(rjn);
  for (uint64_t i = 0; i < num_scs; i++) {
    rjn_node *head = &scs[i].head;
    head->prev = 0;
    head->next = 0;
    head->size = 0;
  }

  rjn_node *first_node = rjn_allocation_unit(rjn, 0);
  first_node->size = rjn->num_allocation_units;
  rjn_node *last_node = rjn_allocation_unit(rjn, rjn->num_allocation_units - 1);
  last_node->size = rjn->num_allocation_units;
  rjn_prepend(rjn, &scs[num_scs - 1], first_node);

  return 0;
}

void rjn_deinit(rjn_allocator *rjn) {
  // Nothing to do.
}

size_t rjn_size(rjn_allocator *rjn) {
  return rjn->num_allocation_units * rjn->allocation_unit_size;
}

size_t rjn_allocation_unit_size(rjn_allocator *rjn) {
  return rjn->allocation_unit_size;
}

void *rjn_start(rjn_allocator *rjn) {
  return rjn_pointer(rjn, rjn->allocation_units_offset);
}

void *rjn_end(rjn_allocator *rjn) {
  return rjn_pointer(rjn,
                     rjn->allocation_units_offset +
                         rjn->num_allocation_units * rjn->allocation_unit_size);
}

void rjn_print_allocation_stats(rjn_allocator *rjn) {
  uint64_t num_sclasses = rjn_num_size_classes(rjn);
  size_class *scs = rjn_size_classes(rjn);

  uint64_t total_free_units = 0;
  uint64_t total_free_chuynks = 0;
  printf("----------------------------------------\n");
  for (uint64_t i = 0; i < num_sclasses; i++) {
    size_class *sc = &scs[i];
    total_free_units += sc->num_units;
    total_free_chuynks += sc->num_chunks;
    printf("size class %2" PRIu64 ": %12" PRIu64 " chunks, %12" PRIu64
           " units\n",
           i, sc->num_chunks, sc->num_units);
  }
  printf("total free chunks: %" PRIu64 "\n", total_free_chuynks);
  printf("total free units: %" PRIu64 "\n", total_free_units);
  printf("total free bytes: %" PRIu64 "\n",
         total_free_units * rjn->allocation_unit_size);
}