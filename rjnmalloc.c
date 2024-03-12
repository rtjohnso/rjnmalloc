#include "rjnmalloc.h"
#include <assert.h>
#include <immintrin.h>
#include <inttypes.h>
#include <string.h>

#define debug_assert(x) assert(x)
//#define debug_assert(x)

struct rjn_allocator {
  size_t region_size;
  size_t allocation_unit_size;
  uint64_t num_allocation_units;
  int track_allocations;
  uint64_t allocated_units;

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
} size_class;

#define MIN_ALLOCATION_UNIT_SIZE (sizeof(rjn_node))

uint64_t rjn_offset(rjn_allocator *rjn, void *ptr) { return ptr - (void *)rjn; }

static void *rjn_pointer(rjn_allocator *rjn, uint64_t offset) {
  return (void *)rjn + offset;
}

static uint8_t *rjn_allocation_units(rjn_allocator *rjn) {
  return (uint8_t *)rjn_pointer(rjn, rjn->allocation_units_offset);
}

static rjn_node *rjn_allocation_unit(rjn_allocator *rjn, uint64_t i) {
  assert(i < rjn->num_allocation_units);
  void *p = rjn_allocation_units(rjn);
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

  int fl = 63 - _lzcnt_u64(units);
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
  // debug_assert(node->next == 0);
  // debug_assert(node->prev == 0);
  node->next = sclass->head.next;
  node->prev = rjn_offset(rjn, sclass);
  if (sclass->head.next) {
    rjn_node *next = rjn_pointer(rjn, sclass->head.next);
    debug_assert(next->prev == rjn_offset(rjn, sclass));
    next->prev = rjn_offset(rjn, node);
  }
  sclass->head.next = rjn_offset(rjn, node);
  __sync_fetch_and_sub(&rjn->allocated_units, node->size);
}

static void rjn_remove(rjn_allocator *rjn, rjn_node *node) {
  rjn_node *prev = rjn_pointer(rjn, node->prev);
  debug_assert(prev->next == rjn_offset(rjn, node));
  prev->next = node->next;
  if (node->next) {
    rjn_node *next = rjn_pointer(rjn, node->next);
    debug_assert(next->prev == rjn_offset(rjn, node));
    next->prev = node->prev;
  }
  __sync_fetch_and_add(&rjn->allocated_units, node->size);
}

#define RJN_META_FREE (0)
#define RJN_META_UNARY (1)
#define RJN_META_BINARY (2)
#define RJN_META_CONTINUATION (3)
#define RJN_MIN_BINARY_SIZE (65)

static int rjn_metadata_cas(rjn_allocator *rjn, uint64_t unit, uint8_t old,
                            uint8_t new) {
  uint8_t *metadata = rjn_metadata_vector(rjn);
  return __sync_bool_compare_and_swap(&metadata[unit], old, new);
}

static void rjn_metadata_set(rjn_allocator *rjn, uint64_t unit, uint8_t value) {
  uint8_t *metadata = rjn_metadata_vector(rjn);
  metadata[unit] = value;
}

static uint8_t rjn_metadata_get(rjn_allocator *rjn, uint64_t unit) {
  uint8_t *metadata = rjn_metadata_vector(rjn);
  return metadata[unit];
}

static int rjn_metadata_cas_start_from_free_to_allocated(rjn_allocator *rjn,
                                                         uint64_t unit,
                                                         uint64_t size) {
  return rjn_metadata_cas(rjn, unit, RJN_META_FREE,
                          RJN_MIN_BINARY_SIZE <= size ? RJN_META_BINARY
                                                      : RJN_META_UNARY);
}

static int rjn_metadata_cas_end_from_free_to_allocated(rjn_allocator *rjn,
                                                       uint64_t unit,
                                                       uint64_t size) {
  return rjn_metadata_cas(rjn, unit, RJN_META_FREE, RJN_META_UNARY);
}

static int rjn_metadata_cas_pred_from_free_to_allocated(rjn_allocator *rjn,
                                                        uint64_t succ) {
  assert(0 < succ);
  return rjn_metadata_cas(rjn, succ - 1, RJN_META_FREE, RJN_META_UNARY);
}

static int rjn_metadata_cas_succ_from_free_to_allocated(rjn_allocator *rjn,
                                                        uint64_t unit,
                                                        uint64_t size) {
  assert(unit + size < rjn->num_allocation_units);
  return rjn_metadata_cas(rjn, unit + size - 1, RJN_META_FREE, RJN_META_UNARY);
}

static void rjn_metadata_set_start_to_allocated(rjn_allocator *rjn,
                                                uint64_t unit, uint64_t size) {
  rjn_metadata_set(rjn, unit,
                   RJN_MIN_BINARY_SIZE <= size ? RJN_META_BINARY
                                               : RJN_META_UNARY);
}

static void rjn_metadata_set_start_to_free(rjn_allocator *rjn, uint64_t start,
                                           uint64_t size) {
  rjn_metadata_set(rjn, start, RJN_META_FREE);
}

static void rjn_metadata_set_end_to_free(rjn_allocator *rjn, uint64_t start,
                                         uint64_t size) {
  rjn_metadata_set(rjn, start + size - 1, RJN_META_FREE);
}

static void rjn_metadata_set_end_to_allocated(rjn_allocator *rjn,
                                              uint64_t start, uint64_t size) {
  if (1 < size) {
    rjn_metadata_set(rjn, start + size - 1, RJN_META_CONTINUATION);
  }
}

static void rjn_metadata_set_size(rjn_allocator *rjn, uint64_t start,
                                  uint64_t size) {
  for (uint64_t i = 1; i < size - 1; i++) {
    rjn_metadata_set(rjn, start + i, RJN_META_CONTINUATION);
  }
}

static uint64_t rjn_metadata_get_size(rjn_allocator *rjn, uint64_t start) {
  uint64_t size = 1;
  while (rjn_metadata_get(rjn, start + size) == RJN_META_CONTINUATION) {
    size++;
  }
  return size;
}

static uint64_t rjn_grab_predecessor_if_free(rjn_allocator *rjn,
                                             uint64_t start) {
  if (start == 0) {
    return 0;
  }
  if (!rjn_metadata_cas_pred_from_free_to_allocated(rjn, start)) {
    return start;
  }
  rjn_node *pred_node = rjn_allocation_unit(rjn, start - 1);
  assert(pred_node->size <= start);
  uint64_t pred_start = start - pred_node->size;
  if (pred_start < start - 1 && !rjn_metadata_cas_start_from_free_to_allocated(
                                    rjn, pred_start, pred_node->size)) {
    rjn_metadata_set_end_to_free(rjn, pred_start, pred_node->size);
    return start;
  }
  return pred_start;
}

static uint64_t rjn_grab_successor_if_free(rjn_allocator *rjn, uint64_t pred,
                                           uint64_t size) {
  if (rjn->num_allocation_units <= pred + size) {
    return 0;
  }
  if (!rjn_metadata_cas_succ_from_free_to_allocated(rjn, pred, size)) {
    return 0;
  }
  rjn_node *succ_node = rjn_allocation_unit(rjn, pred + size);
  uint64_t succ_end = pred + size + succ_node->size - 1;
  if (pred + size < succ_end) {
    while (!rjn_metadata_cas_end_from_free_to_allocated(rjn, pred + size,
                                                        succ_node->size)) {
      _mm_pause();
    }
  }
  return pred + size;
}

static void rjn_free_chunk(rjn_allocator *rjn, uint64_t start, uint64_t size) {
  rjn_node *start_node;
restart:
  start_node = rjn_allocation_unit(rjn, start);
  rjn_node *end_node = rjn_allocation_unit(rjn, start + size - 1);

  start_node->size = size;
  end_node->size = size;

  size_class *sc = rjn_find_size_class(rjn, size);
  rjn_lock_size_class(sc);
  rjn_prepend(rjn, sc, start_node);

  rjn_metadata_set_start_to_free(rjn, start, size);

  uint64_t new_start = rjn_grab_predecessor_if_free(rjn, start);
  if (new_start < start) {
    rjn_remove(rjn, start_node);
    rjn_unlock_size_class(sc);
    rjn_node *pred_start_node = rjn_allocation_unit(rjn, new_start);
    size_class *pred_sc = rjn_find_size_class(rjn, pred_start_node->size);
    rjn_lock_size_class(pred_sc);
    rjn_remove(rjn, pred_start_node);
    rjn_unlock_size_class(pred_sc);
    start = new_start;
    size += pred_start_node->size;
    goto restart;
  }

  rjn_metadata_set_end_to_free(rjn, start, size);
  rjn_unlock_size_class(sc);

  uint64_t successor_start = rjn_grab_successor_if_free(rjn, start, size);
  if (successor_start) {
    rjn_node *succ_start_node = rjn_allocation_unit(rjn, successor_start);
    size_class *succ_sc = rjn_find_size_class(rjn, succ_start_node->size);
    rjn_lock_size_class(succ_sc);
    rjn_remove(rjn, succ_start_node);
    rjn_unlock_size_class(succ_sc);
    start = successor_start;
    size = succ_start_node->size;
    goto restart;
  }
}

static void *rjn_alloc_from_size_class(rjn_allocator *rjn, unsigned int scidx,
                                       size_t alignment_units, size_t units) {
  size_class *sc = &rjn_size_classes(rjn)[scidx];
  rjn_lock_size_class(sc);
  for (uint64_t curr_off = sc->head.next; curr_off;
       curr_off = ((rjn_node *)rjn_pointer(rjn, curr_off))->next) {
    uint64_t first_unit =
        (curr_off - rjn->allocation_units_offset) / rjn->allocation_unit_size;
    if (!rjn_metadata_cas_start_from_free_to_allocated(rjn, first_unit,
                                                       units)) {
      // If the CAS fails, that means that someone else is freeing the
      // previous chunk and merging their free chunk with the current one.
      continue;
    }
    rjn_node *node = rjn_pointer(rjn, curr_off);

    // Check if the current chunk is large enough to satisfy the request,
    // including alignment padding.
    uint64_t pad_units = 0;
    if (alignment_units) {
      pad_units =
          (alignment_units - (first_unit % alignment_units)) % alignment_units;
    }
    if (node->size < pad_units + units) {
      rjn_metadata_set_start_to_free(rjn, first_unit, units);
      continue;
    }

    // Allocate the chunk.
    rjn_metadata_set_end_to_allocated(rjn, first_unit, node->size);
    rjn_remove(rjn, node);
    rjn_unlock_size_class(sc);

    // Give back any alignment pad at the beginning
    if (pad_units) {
      rjn_metadata_set_end_to_allocated(rjn, first_unit, pad_units);
      rjn_metadata_set_start_to_allocated(rjn, first_unit + pad_units, units);
      rjn_free_chunk(rjn, first_unit, pad_units);
    }

    // Give back any remaining space at the end
    if (pad_units + units < node->size) {
      if (1 < units) {
        rjn_metadata_set_end_to_allocated(rjn, first_unit + pad_units, units);
      }
      rjn_metadata_set_start_to_allocated(rjn, first_unit + pad_units + units,
                                          node->size - pad_units - units);
      rjn_free_chunk(rjn, first_unit + units, node->size - units);
    }

    rjn_metadata_set_size(rjn, first_unit + pad_units, units);

    return node;
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

  size_t units = (size + rjn->allocation_unit_size - 1) /
                 rjn->allocation_unit_size; // round up
  size_t alignment_units = alignment / rjn->allocation_unit_size;
  int num_sclasses = rjn_num_size_classes(rjn);

  // First search the larger size classes.  Any node in any of those size
  // classes should be sufficient to satisfy the request (except possibly due to
  // alignment constraints).
  for (unsigned int scidx = 1 + rjn_find_size_class_index(rjn, units);
       scidx < num_sclasses; scidx++) {
    void *ptr = rjn_alloc_from_size_class(rjn, scidx, alignment_units, units);
    if (ptr) {
      return ptr;
    }
  }
  // OK we're desperate.  Try groveling through the size class for this
  // particular size to see if there's anything there.
  return rjn_alloc_from_size_class(rjn, rjn_find_size_class_index(rjn, units),
                                   alignment_units, units);
}

void rjn_free(rjn_allocator *rjn, void *ptr) {
  uint64_t first_unit = (rjn_offset(rjn, ptr) - rjn->allocation_units_offset) /
                        rjn->allocation_unit_size;
  uint64_t size = rjn_metadata_get_size(rjn, first_unit);
  //  memset(ptr, 0, size * rjn->allocation_unit_size);
  rjn_free_chunk(rjn, first_unit, size);
}

int rjn_init(rjn_allocator *rjn, size_t region_size,
             size_t allocation_unit_size, int track_allocations) {
  if (region_size < sizeof(rjn_allocator)) {
    return -1;
  }
  if (allocation_unit_size < MIN_ALLOCATION_UNIT_SIZE) {
    return -1;
  }

  rjn->region_size = region_size;
  rjn->allocation_unit_size = allocation_unit_size;
  rjn->track_allocations = track_allocations;

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

  rjn->allocated_units = 0;

  return 0;
}

void rjn_deinit(rjn_allocator *rjn) {
  // Nothing to do.
}

size_t rjn_size(rjn_allocator *rjn) {
  return rjn->num_allocation_units * rjn->allocation_unit_size;
}

void *rjn_start(rjn_allocator *rjn) {
  return rjn_pointer(rjn, rjn->allocation_units_offset);
}

void *rjn_end(rjn_allocator *rjn) {
  return rjn_pointer(rjn,
                     rjn->allocation_units_offset +
                         rjn->num_allocation_units * rjn->allocation_unit_size);
}

size_t rjn_allocated(rjn_allocator *rjn) {
  assert(rjn->track_allocations);
  return rjn->allocated_units * rjn->allocation_unit_size;
}