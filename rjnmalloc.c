#include "rjnmalloc.h"
#include <assert.h>
#include <immintrin.h>
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#define RJN_DEBUG 0
#define RJN_DUMP_CHUNKS 1
#define debug_code __attribute__((unused))
#if RJN_DEBUG
#define debug_printf(...) printf(__VA_ARGS__)
#else
#define debug_printf(...)
#endif

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

#define RJN_META_CONTINUATION (0)
#define RJN_META_UNARY (1)
#define RJN_META_BINARY (2)
#define RJN_META_FREE (3)
#define BITS_PER_META_ENTRY (2)
#define RJN_MIN_BINARY_UNITS                                                   \
  (2 + (8 - BITS_PER_META_ENTRY) / BITS_PER_META_ENTRY +                       \
   8 * sizeof(uint64_t) / BITS_PER_META_ENTRY)

static const char *metaname[] = {"C", "U", "B", "F"};

/*
 * Elementary operations
 */
uint64_t rjn_offset(rjn_allocator *rjn, void *ptr) {
  assert((uint8_t *)ptr >= (uint8_t *)rjn);
  assert((uint8_t *)ptr < (uint8_t *)rjn + rjn->region_size);
  return (uint8_t *)ptr - (uint8_t *)rjn;
}

static void *rjn_pointer(rjn_allocator *rjn, uint64_t offset) {
  assert(offset < rjn->region_size);
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

/*
 * Metadata operations
 */

#define META_ENTRIES_PER_BYTE (8 / BITS_PER_META_ENTRY)
#define META_ENTRY_MASK ((1 << BITS_PER_META_ENTRY) - 1)
#define VECTOR_INDEX(i) ((i) / META_ENTRIES_PER_BYTE)
#define VECTOR_BYTE(v, i) ((v)[VECTOR_INDEX(i)])
#define BYTE_EXTRACT(b, i)                                                     \
  (((b) >> (BITS_PER_META_ENTRY * ((i) % META_ENTRIES_PER_BYTE))) &            \
   META_ENTRY_MASK)
#define META_GET(m, u) BYTE_EXTRACT(VECTOR_BYTE(m, u), u)
#define BYTE_INSERT(b, i)                                                      \
  (((b)&META_ENTRY_MASK) << (BITS_PER_META_ENTRY *                             \
                             ((i) % META_ENTRIES_PER_BYTE)))
#define META_CONTINUATION_BYTE (RJN_META_CONTINUATION * 0xcc)

static uint8_t *rjn_metadata_vector(rjn_allocator *rjn) {
  return (uint8_t *)rjn_pointer(rjn, rjn->metadata_offset);
}

static uint8_t rjn_metadata_get(rjn_allocator *rjn, uint64_t unit) {
  assert(unit < rjn->num_allocation_units);
  uint8_t *metadata = rjn_metadata_vector(rjn);
  return META_GET(metadata, unit);
}

static int rjn_metadata_cas(rjn_allocator *rjn, uint64_t unit, uint8_t old,
                            uint8_t new) {
  assert(unit < rjn->num_allocation_units);
  uint8_t *metadata = rjn_metadata_vector(rjn);
  uint8_t mask = BYTE_INSERT(old ^ new, unit);
  uint8_t oldbyte;
  do {
    oldbyte = VECTOR_BYTE(metadata, unit);
    if (BYTE_EXTRACT(oldbyte, unit) != old) {
      return 0;
    }
  } while (!__sync_bool_compare_and_swap(&VECTOR_BYTE(metadata, unit), oldbyte,
                                         oldbyte ^ mask));
  return 1;
}

static void rjn_metadata_set(rjn_allocator *rjn, uint64_t unit, uint8_t value) {
  assert(unit < rjn->num_allocation_units);
  uint8_t *metadata = rjn_metadata_vector(rjn);
  debug_printf("metadata_set %lu %s %s\n", unit,
               META_GET(metadata, unit) < 4 ? metaname[META_GET(metadata, unit)]
                                            : "size?",
               metaname[value]);

  uint8_t oldbyte;
  uint8_t newbyte;
  do {
    oldbyte = VECTOR_BYTE(metadata, unit);
    newbyte = oldbyte ^ BYTE_INSERT(value ^ BYTE_EXTRACT(oldbyte, unit), unit);
  } while (!__sync_bool_compare_and_swap(&VECTOR_BYTE(metadata, unit), oldbyte,
                                         newbyte));
}

/* This performs the following logical operation atomically:
 * if unit == rjn->num_allocation_units - 1 or
      metadata[uniit + 1] == RJN_META_UNARY or
      metadata[unit + 1] == RJN_META_BINARY
 * then
     set metadata[unit] to RJN_META_FREE
     return 1
 * else
     return 0
 */
static int rjn_metadata_try_set_end_to_free(rjn_allocator *rjn, uint64_t unit) {
  assert(unit < rjn->num_allocation_units);
  if (unit == rjn->num_allocation_units - 1) {
    rjn_metadata_set(rjn, unit, RJN_META_FREE);
    return 1;
  }

  uint8_t *metadata = rjn_metadata_vector(rjn);
  uint64_t moff = VECTOR_INDEX(unit);
  uint8_t oldmeta[2] = {metadata[moff], metadata[moff + 1]};
  uint8_t *oldmetap = &oldmeta[0] - moff;
  uint8_t oldme = META_GET(oldmetap, unit);
  uint8_t oldsucc = META_GET(oldmetap, unit + 1);
  assert(oldme != RJN_META_FREE);
  assert(oldsucc != RJN_META_CONTINUATION);
  if (oldsucc == RJN_META_FREE) {
    return 0;
  }
  uint8_t newmeta[2] = {oldmeta[0] ^ BYTE_INSERT(oldme ^ RJN_META_FREE, unit),
                        oldmeta[1]};
  if (VECTOR_INDEX(unit + 1) == VECTOR_INDEX(unit)) {
    // Do a 1-byte CAS, which reduces the probability of performing a
    // cross-cache-line CAS, which are hella slow.
    return __sync_bool_compare_and_swap(&metadata[moff], oldmeta[0],
                                        newmeta[0]);
  } else {
    uint16_t *p = (uint16_t *)&metadata[moff];
    return __sync_bool_compare_and_swap(p, *(uint16_t *)oldmeta,
                                        *(uint16_t *)newmeta);
  }
}

/* This performs the following logical operation atomically
 * if unit == 0 or
      metadata[uniit - 1] != RJN_META_FREE
 * then
     set metadata[unit] to RJN_META_FREE
     return 1
 * else
     return 0
 */
static int rjn_metadata_try_set_start_to_free(rjn_allocator *rjn,
                                              uint64_t unit) {
  assert(unit < rjn->num_allocation_units);
  if (unit == 0) {
    rjn_metadata_set(rjn, unit, RJN_META_FREE);
    return 1;
  }

  uint8_t *metadata = rjn_metadata_vector(rjn);
  uint64_t moff = VECTOR_INDEX(unit - 1);
  uint8_t oldmeta[2] = {metadata[moff], metadata[moff + 1]};
  uint8_t *oldmetap = &oldmeta[0] - moff;
  uint8_t oldpred = META_GET(oldmetap, unit - 1);
  uint8_t oldme = META_GET(oldmetap, unit);
  assert(oldme != RJN_META_FREE);
  if (oldpred == RJN_META_FREE) {
    return 0;
  }
  uint8_t newmeta[2] = {oldmeta[0], oldmeta[1]};
  uint8_t *newmetap = &newmeta[0] - moff;
  newmetap[VECTOR_INDEX(unit)] ^= BYTE_INSERT(oldme ^ RJN_META_FREE, unit);
  if (VECTOR_INDEX(unit) == moff) {
    // Do a 1-byte CAS, which reduces the probability of performing a
    // cross-cache-line CAS, which are hella slow.
    return __sync_bool_compare_and_swap(&metadata[moff], oldmeta[0],
                                        newmeta[0]);
  } else {
    uint16_t *p = (uint16_t *)&metadata[moff];
    return __sync_bool_compare_and_swap(p, *(uint16_t *)oldmeta,
                                        *(uint16_t *)newmeta);
  }
}

/* This performs the following logical operation atomically
 * if (unit == 0 or
      metadata[uniit - 1] == RJN_META_UNARY or
      metadata[unit - 1] == RJN_META_BINARY)
      and
      (unit == rjn->num_allocation_units - 1 or
      metadata[uniit + 1] == RJN_META_UNARY or
      metadata[unit + 1] == RJN_META_BINARY)
 * then
     set metadata[unit] to RJN_META_FREE
     return 1
 * else
     return 0
 */
static int rjn_metadata_try_set_singleton_to_free(rjn_allocator *rjn,
                                                  uint64_t unit) {
  assert(unit < rjn->num_allocation_units);
  if (unit == 0 && unit == rjn->num_allocation_units - 1) {
    rjn_metadata_set(rjn, unit, RJN_META_FREE);
    return 1;
  } else if (unit == 0) {
    return rjn_metadata_try_set_end_to_free(rjn, unit);
  } else if (unit == rjn->num_allocation_units - 1) {
    return rjn_metadata_try_set_start_to_free(rjn, unit);
  } else {
  }

  uint8_t *metadata = rjn_metadata_vector(rjn);
  uint64_t moff = VECTOR_INDEX(unit - 1);
  uint8_t oldmeta[2] = {metadata[moff], metadata[moff + 1]};
  uint8_t *oldmetap = &oldmeta[0] - moff;
  uint8_t oldpred = META_GET(oldmetap, unit - 1);
  uint8_t oldme = META_GET(oldmetap, unit);
  uint8_t oldsucc = META_GET(oldmetap, unit + 1);
  assert(oldme != RJN_META_FREE);
  assert(oldsucc != RJN_META_CONTINUATION);
  if (oldpred == RJN_META_FREE || oldsucc == RJN_META_FREE) {
    return 0;
  }
  uint8_t newmeta[2] = {oldmeta[0], oldmeta[1]};
  uint8_t *newmetap = &newmeta[0] - moff;
  newmetap[VECTOR_INDEX(unit)] ^= BYTE_INSERT(oldme ^ RJN_META_FREE, unit);
  if (VECTOR_INDEX(unit + 1) == VECTOR_INDEX(unit - 1)) {
    // Do a 1-byte CAS, which reduces the probability of performing a
    // cross-cache-line CAS, which are hella slow.
    return __sync_bool_compare_and_swap(&metadata[moff], oldmeta[0],
                                        newmeta[0]);
  } else {
    uint16_t *p = (uint16_t *)&metadata[moff];
    return __sync_bool_compare_and_swap(p, *(uint16_t *)oldmeta,
                                        *(uint16_t *)newmeta);
  }
}

static void rjn_metadata_set_size(rjn_allocator *rjn, uint64_t first_unit,
                                  uint64_t units) {
  assert(first_unit + units <= rjn->num_allocation_units);
  uint8_t *metadata = rjn_metadata_vector(rjn);
  if (units < RJN_MIN_BINARY_UNITS) {
    assert(META_GET(metadata, first_unit) == RJN_META_UNARY);
    assert(first_unit + units <= rjn->num_allocation_units);
    uint64_t i;
    for (i = 1; i < units - 1 && (first_unit + i) % META_ENTRIES_PER_BYTE;
         i++) {
      rjn_metadata_set(rjn, first_unit + i, RJN_META_CONTINUATION);
    }
    memset(&metadata[VECTOR_INDEX(first_unit + i)], META_CONTINUATION_BYTE,
           (units - i) / META_ENTRIES_PER_BYTE);
    for (i = 1; i < units - 1; i++) {
      rjn_metadata_set(rjn, first_unit + i, RJN_META_CONTINUATION);
    }
  } else {
    assert(META_GET(metadata, first_unit) != RJN_META_FREE);
    uint64_t size_start =
        (first_unit + META_ENTRIES_PER_BYTE) / META_ENTRIES_PER_BYTE;
    assert(size_start + sizeof(uint64_t) <=
           (first_unit + units) / META_ENTRIES_PER_BYTE);
    rjn_metadata_set(rjn, first_unit, RJN_META_BINARY);
    memcpy(&metadata[size_start], &units, sizeof(uint64_t));
  }
}

static uint64_t rjn_metadata_get_size(rjn_allocator *rjn, uint64_t first_unit) {
  assert(first_unit < rjn->num_allocation_units);
  uint8_t *metadata = rjn_metadata_vector(rjn);
  uint64_t size;
  if (META_GET(metadata, first_unit) == RJN_META_UNARY) {
    size = 1;
    while (first_unit + size < rjn->num_allocation_units &&
           (first_unit + size) % META_ENTRIES_PER_BYTE &&
           META_GET(metadata, first_unit + size) == RJN_META_CONTINUATION) {
      size++;
    }
    while (first_unit + size + META_ENTRIES_PER_BYTE <=
               rjn->num_allocation_units &&
           metadata[VECTOR_INDEX(first_unit + size)] ==
               META_CONTINUATION_BYTE) {
      size += META_ENTRIES_PER_BYTE;
    }
    while (first_unit + size < rjn->num_allocation_units &&
           META_GET(metadata, first_unit + size) == RJN_META_CONTINUATION) {
      size++;
    }
  } else {
    assert(first_unit + RJN_MIN_BINARY_UNITS <= rjn->num_allocation_units);
    memcpy(
        &size,
        &metadata[(first_unit + META_ENTRIES_PER_BYTE) / META_ENTRIES_PER_BYTE],
        sizeof(uint64_t));
  }
  return size;
}

static void rjn_metadata_erase_size(rjn_allocator *rjn, uint64_t first_unit,
                                    uint64_t units) {
  assert(first_unit + units <= rjn->num_allocation_units);
  uint8_t *metadata = rjn_metadata_vector(rjn);
  if (units < RJN_MIN_BINARY_UNITS) {
    assert(META_GET(metadata, first_unit) == RJN_META_UNARY);
    uint64_t i;
    for (i = 1; i < units - 1 && (first_unit + i) % META_ENTRIES_PER_BYTE;
         i++) {
      rjn_metadata_set(rjn, first_unit + i, RJN_META_CONTINUATION);
    }
    while (i + META_ENTRIES_PER_BYTE <= units - 1) {
      metadata[(first_unit + i) / META_ENTRIES_PER_BYTE] =
          META_CONTINUATION_BYTE;
      i += META_ENTRIES_PER_BYTE;
    }
    while (i < units - 1) {
      rjn_metadata_set(rjn, first_unit + i, RJN_META_CONTINUATION);
      i++;
    }
  } else {
    memset(
        &metadata[(first_unit + META_ENTRIES_PER_BYTE) / META_ENTRIES_PER_BYTE],
        META_CONTINUATION_BYTE, sizeof(uint64_t));
  }
}

/*
 * Debug printing code
 */
debug_code static void node_to_string(rjn_allocator *rjn, rjn_node *node,
                                      char *buffer, size_t buffer_size) {
  uint64_t offset = rjn_offset(rjn, node);
  assert((offset - rjn->allocation_units_offset) % rjn->allocation_unit_size ==
         0);
  uint64_t aunum =
      (offset - rjn->allocation_units_offset) / rjn->allocation_unit_size;
  assert(aunum < rjn->num_allocation_units);

  snprintf(buffer, buffer_size,
           "    node %p\n"
           "    offset %lu\n"
           "    aunum %lu\n"
           "    prev %lu\n"
           "    next %lu\n"
           "    size %lu\n"
           "    %s %s",
           (void *)node, offset, aunum, node->prev, node->next, node->size,
           metaname[rjn_metadata_get(rjn, aunum)],
           0 < node->size
               ? metaname[rjn_metadata_get(rjn, aunum + node->size - 1)]
               : "-");
}

#define node_string(rjn, node)                                                 \
  (({                                                                          \
     struct {                                                                  \
       char buffer[256];                                                       \
     } b;                                                                      \
     node_to_string((rjn), (node), b.buffer, sizeof(b.buffer));                \
     b;                                                                        \
   }).buffer)

#define offset_string(rjn, off)                                                \
  (({                                                                          \
     struct {                                                                  \
       char buffer[256];                                                       \
     } b;                                                                      \
     if (off) {                                                                \
       node_to_string((rjn), rjn_pointer(rjn, off), b.buffer,                  \
                      sizeof(b.buffer));                                       \
     } else {                                                                  \
       snprintf(b.buffer, sizeof(b.buffer), "    (null)");                     \
     }                                                                         \
     b;                                                                        \
   }).buffer)

#define aunum_string(rjn, aunum)                                               \
  (({                                                                          \
     struct {                                                                  \
       char buffer[256];                                                       \
     } b;                                                                      \
     node_to_string((rjn), rjn_allocation_unit(rjn, aunum), b.buffer,          \
                    sizeof(b.buffer));                                         \
     b;                                                                        \
   }).buffer)

/*
 * Size-class operations
 */

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

  assert(units);
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
  assert((void *)node != (void *)rjn);
  assert(node->next == 0);
  assert(node->prev == 0);
  node->next = sclass->head.next;
  node->prev = rjn_offset(rjn, sclass);
  if (sclass->head.next) {
    rjn_node *next = rjn_pointer(rjn, sclass->head.next);
    assert(next != node);
    assert(next->prev == rjn_offset(rjn, sclass));
    next->prev = rjn_offset(rjn, node);
  }
  sclass->head.next = rjn_offset(rjn, node);
  sclass->num_chunks++;
  sclass->num_units += node->size;
}

static void rjn_remove(rjn_allocator *rjn, size_class *sclass, rjn_node *node) {
  assert((void *)node != (void *)rjn);
  assert(node->prev != 0);
  rjn_node *prev = rjn_pointer(rjn, node->prev);
  assert(prev->next == rjn_offset(rjn, node));
  prev->next = node->next;
  if (node->next) {
    rjn_node *next = rjn_pointer(rjn, node->next);
    assert(next->prev == rjn_offset(rjn, node));
    next->prev = node->prev;
  }
  node->next = node->prev = 0;
  sclass->num_chunks--;
  sclass->num_units -= node->size;
}

/*
 * Debugging validation code
 */

typedef void (*chunk_walk_func)(rjn_allocator *rjn, uint64_t start_unit,
                                uint64_t size, void *arg);

static void rjn_walk_chunks(rjn_allocator *rjn, chunk_walk_func func,
                            void *arg) {
  uint64_t curr_unit = 0;
  while (curr_unit < rjn->num_allocation_units) {
    uint64_t size = 0;
    switch (rjn_metadata_get(rjn, curr_unit)) {
    case RJN_META_FREE:
      rjn_node *node = rjn_allocation_unit(rjn, curr_unit);
      size = node->size;
      break;
    case RJN_META_UNARY:
    case RJN_META_BINARY:
      size = rjn_metadata_get_size(rjn, curr_unit);
      break;
    default:
      assert(0);
      break;
    }
    assert(size);
    if (1 < RJN_DEBUG) {
      for (uint64_t i = rjn_metadata_get(rjn, curr_unit) == RJN_META_BINARY
                            ? 1 + sizeof(uint64_t)
                            : 1;
           i < size - 1; i++) {
        assert(rjn_metadata_get(rjn, curr_unit + i) == RJN_META_CONTINUATION);
      }
    }
    if (1 < size) {
      if (rjn_metadata_get(rjn, curr_unit) == RJN_META_FREE) {
        assert(rjn_metadata_get(rjn, curr_unit + size - 1) == RJN_META_FREE);
      } else {
        assert(rjn_metadata_get(rjn, curr_unit + size - 1) ==
               RJN_META_CONTINUATION);
      }
    }
    func(rjn, curr_unit, size, arg);
    curr_unit += size;
    assert(curr_unit <= rjn->num_allocation_units);
  }
}

static void rjn_count_chunks(rjn_allocator *rjn, uint64_t start_unit,
                             uint64_t size, void *arg) {
  uint64_t *count = (uint64_t *)arg;
  (*count)++;
}

typedef struct chunk {
  uint64_t start;
  uint64_t units;
  size_class *sc;
} chunk;

typedef struct chunk_array {
  uint64_t num_chunks;
  uint64_t max_chunks;
  chunk chunks[];
} chunk_array;

static int compare_chunk_start(const void *a, const void *b) {
  const chunk *ca = (const chunk *)a;
  const chunk *cb = (const chunk *)b;
  if (ca->start < cb->start) {
    return -1;
  } else if (ca->start > cb->start) {
    return 1;
  } else {
    return 0;
  }
}

static uint64_t rjn_find_chunk_in_array(chunk_array *ca, uint64_t start) {
  chunk key = {.start = start};
  void *p = bsearch(&key, ca->chunks, ca->num_chunks, sizeof(chunk),
                    compare_chunk_start);
  assert(p);
  uint64_t i = (chunk *)p - ca->chunks;
  assert(i < ca->num_chunks);
  assert(ca->chunks[i].start == start);
  return i;
}

static void rjn_collect_chunks(rjn_allocator *rjn, uint64_t start_unit,
                               uint64_t size, void *arg) {
  chunk_array *ca = (chunk_array *)arg;
  assert(ca->num_chunks < ca->max_chunks);
  ca->chunks[ca->num_chunks].start = start_unit;
  ca->chunks[ca->num_chunks].units = size;
  ca->chunks[ca->num_chunks].sc = NULL;
  ca->num_chunks++;
}

static void rjn_validate_size_classes(rjn_allocator *rjn, chunk_array *ca) {
  uint64_t num_sclasses = rjn_num_size_classes(rjn);
  size_class *scs = rjn_size_classes(rjn);
  for (uint64_t i = 0; i < num_sclasses; i++) {
    size_class *sc = &scs[i];
    for (uint64_t curr_off = sc->head.next; curr_off;
         curr_off = ((rjn_node *)rjn_pointer(rjn, curr_off))->next) {
      uint64_t start_unit =
          (curr_off - rjn->allocation_units_offset) / rjn->allocation_unit_size;
      uint64_t c = rjn_find_chunk_in_array(ca, start_unit);
      assert(ca->chunks[c].sc == NULL);
      ca->chunks[c].sc = sc;
    }
  }

  for (uint64_t i = 0; i < ca->num_chunks; i++) {
    if (rjn_metadata_get(rjn, ca->chunks[i].start) == RJN_META_FREE) {
      assert(ca->chunks[i].sc);
    }
  }
}

debug_code static void rjn_validate(rjn_allocator *rjn) {
  if (RJN_DEBUG) {
    uint64_t num_chunks = 0;
    rjn_walk_chunks(rjn, rjn_count_chunks, &num_chunks);
    chunk_array *ca = malloc(sizeof(chunk_array) + num_chunks * sizeof(chunk));
    ca->num_chunks = 0;
    ca->max_chunks = num_chunks;
    rjn_walk_chunks(rjn, rjn_collect_chunks, ca);
    rjn_validate_size_classes(rjn, ca);
    if (RJN_DUMP_CHUNKS) {
      printf("chunks\n");
      for (uint64_t i = 0; i < ca->num_chunks; i++) {
        printf("  chunk %12lu %12lu %s %s\n", ca->chunks[i].start,
               ca->chunks[i].units,
               metaname[rjn_metadata_get(rjn, ca->chunks[i].start)],
               1 < ca->chunks[i].units
                   ? metaname[rjn_metadata_get(
                         rjn, ca->chunks[i].start + ca->chunks[i].units - 1)]
                   : "");
      }
    }
    free(ca);
  }
}

/*
 * Chunk freeing functions
 */

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
  if (pred_start < pred_last) {
    if (!rjn_metadata_cas(rjn, pred_start, RJN_META_FREE, RJN_META_UNARY)) {
      rjn_metadata_set(rjn, pred_last, RJN_META_FREE);
      return my_start;
    }
    rjn_metadata_set(rjn, pred_last, RJN_META_CONTINUATION);
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
  uint64_t succ_last = succ_start + succ_node->size - 1;
  if (succ_start < succ_last) {
    while (!rjn_metadata_cas(rjn, succ_last, RJN_META_FREE,
                             RJN_META_CONTINUATION)) {
      _mm_pause();
    }
  }
  return succ_start;
}

static void rjn_free_chunk(rjn_allocator *rjn, uint64_t start_unit,
                           uint64_t units) {
  rjn_node *start_node;

restart:
  debug_printf("free_chunk %lu %lu\n", start_unit, units);

  assert(rjn_metadata_get(rjn, start_unit) == RJN_META_UNARY ||
         rjn_metadata_get(rjn, start_unit) == RJN_META_BINARY);
  if (1 < units) {
    assert(rjn_metadata_get(rjn, start_unit + units - 1) ==
           RJN_META_CONTINUATION);
  }

  rjn_metadata_erase_size(rjn, start_unit, units);

  start_node = rjn_allocation_unit(rjn, start_unit);
  rjn_node *last_node = rjn_allocation_unit(rjn, start_unit + units - 1);

  start_node->size = units;
  last_node->size = units;

  size_class *sc = rjn_find_size_class(rjn, units);
  rjn_lock_size_class(sc);
  rjn_prepend(rjn, sc, start_node);
  rjn_unlock_size_class(sc);

  if (units == 1) {

    if (!rjn_metadata_try_set_singleton_to_free(rjn, start_unit)) {
      rjn_lock_size_class(sc);
      rjn_remove(rjn, sc, start_node);
      rjn_unlock_size_class(sc);

      // Try to merge with our predecessor.
      uint64_t new_start = rjn_grab_predecessor_if_free(rjn, start_unit);
      if (new_start < start_unit) {
        rjn_node *pred_start_node = rjn_allocation_unit(rjn, new_start);
        debug_printf("free_chunk merging %lu %lu with %lu %lu\n", start_unit,
                     units, new_start, pred_start_node->size);
        size_class *pred_sc = rjn_find_size_class(rjn, pred_start_node->size);
        rjn_lock_size_class(pred_sc);
        rjn_remove(rjn, pred_sc, pred_start_node);
        rjn_unlock_size_class(pred_sc);
        rjn_metadata_set(rjn, start_unit, RJN_META_CONTINUATION);
        start_unit = new_start;
        units += pred_start_node->size;
      }

      // Try to merge with our successor.
      uint64_t successor_start =
          rjn_grab_successor_if_free(rjn, start_unit, units);
      if (successor_start) {
        rjn_node *succ_start_node = rjn_allocation_unit(rjn, successor_start);
        debug_printf("free_chunk merging %lu %lu with %lu %lu\n", start_unit,
                     units, successor_start, succ_start_node->size);
        size_class *succ_sc = rjn_find_size_class(rjn, succ_start_node->size);
        rjn_lock_size_class(succ_sc);
        rjn_remove(rjn, succ_sc, succ_start_node);
        rjn_unlock_size_class(succ_sc);
        rjn_metadata_set(rjn, successor_start, RJN_META_CONTINUATION);
        units += succ_start_node->size;
      }

      goto restart;
    }

  } else {

    if (!rjn_metadata_try_set_start_to_free(rjn, start_unit)) {
      rjn_lock_size_class(sc);
      rjn_remove(rjn, sc, start_node);
      rjn_unlock_size_class(sc);

      // Try to merge with our predecessor.
      uint64_t new_start = rjn_grab_predecessor_if_free(rjn, start_unit);
      if (new_start < start_unit) {
        rjn_node *pred_start_node = rjn_allocation_unit(rjn, new_start);
        debug_printf("free_chunk merging %lu %lu with %lu %lu\n", start_unit,
                     units, new_start, pred_start_node->size);
        size_class *pred_sc = rjn_find_size_class(rjn, pred_start_node->size);
        rjn_lock_size_class(pred_sc);
        rjn_remove(rjn, pred_sc, pred_start_node);
        rjn_unlock_size_class(pred_sc);
        rjn_metadata_set(rjn, start_unit, RJN_META_CONTINUATION);
        start_unit = new_start;
        units += pred_start_node->size;
      }

      goto restart;
    }

    if (!rjn_metadata_try_set_end_to_free(rjn, start_unit + units - 1)) {
      if (!rjn_metadata_cas(rjn, start_unit, RJN_META_FREE, RJN_META_UNARY)) {
        // Someone else is already trying to allocate this chunk or to merge
        // with this chunk from the left, so we can finish freeing the chunk and
        // return.
        rjn_metadata_set(rjn, start_unit + units - 1, RJN_META_FREE);
      } else {
        rjn_lock_size_class(sc);
        rjn_remove(rjn, sc, start_node);
        rjn_unlock_size_class(sc);

        // Try to merge with our successor.
        uint64_t successor_start =
            rjn_grab_successor_if_free(rjn, start_unit, units);
        if (successor_start) {
          rjn_node *succ_start_node = rjn_allocation_unit(rjn, successor_start);
          debug_printf("free_chunk merging %lu %lu with %lu %lu\n", start_unit,
                       units, successor_start, succ_start_node->size);
          size_class *succ_sc = rjn_find_size_class(rjn, succ_start_node->size);
          rjn_lock_size_class(succ_sc);
          rjn_remove(rjn, succ_sc, succ_start_node);
          rjn_unlock_size_class(succ_sc);
          rjn_metadata_set(rjn, successor_start, RJN_META_CONTINUATION);
          units += succ_start_node->size;
        }

        goto restart;
      }
    }
  }
}

void rjn_free(rjn_allocator *rjn, void *ptr) {
  rjn_validate(rjn);
  uint64_t first_unit = (rjn_offset(rjn, ptr) - rjn->allocation_units_offset) /
                        rjn->allocation_unit_size;
  uint64_t size = rjn_metadata_get_size(rjn, first_unit);
  assert(size);
  assert(size <= rjn->num_allocation_units);
  assert(rjn_metadata_get(rjn, first_unit) == RJN_META_UNARY ||
         rjn_metadata_get(rjn, first_unit) == RJN_META_BINARY);
  assert(size == 1 ||
         rjn_metadata_get(rjn, first_unit + size - 1) == RJN_META_CONTINUATION);
  rjn_node *node = rjn_allocation_unit(rjn, first_unit);
  node->prev = node->next = 0;
  rjn_free_chunk(rjn, first_unit, size);
  rjn_validate(rjn);
}

/*
 * Allocation functions
 */

static void *rjn_alloc_from_size_class(rjn_allocator *rjn, unsigned int scidx,
                                       size_t alignment_bytes, size_t bytes) {
restart:
  rjn_validate(rjn);

  size_class *sc = &rjn_size_classes(rjn)[scidx];
  if (!sc->num_chunks) {
    return NULL;
  }
  rjn_lock_size_class(sc);

  uint64_t next_off;
  for (uint64_t curr_off = sc->head.next; curr_off; curr_off = next_off) {
    next_off = ((rjn_node *)rjn_pointer(rjn, curr_off))->next;

    uint64_t first_unit =
        (curr_off - rjn->allocation_units_offset) / rjn->allocation_unit_size;

    assert(rjn_metadata_get(rjn, first_unit) != RJN_META_CONTINUATION);
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
      if (rjn_metadata_try_set_start_to_free(rjn, first_unit)) {
        continue;
      }
      // Aw crud.  Someone freed the preceding chunk, so we can't just quietly
      // put this one back.  We have to merge it with the preceding chunk.
      // Merging chunks can require interacting with lots of different size
      // classes (because merges can cascade), so we're just gonna give up our
      // lock on this size class and restart from scratch.
      if (1 < node->size) {
        while (!rjn_metadata_cas(rjn, first_unit + node->size - 1,
                                 RJN_META_FREE, RJN_META_CONTINUATION)) {
          _mm_pause();
        }
      }
      rjn_remove(rjn, sc, node);
      rjn_unlock_size_class(sc);
      rjn_free_chunk(rjn, first_unit, node->size);
      goto restart;
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
    uint64_t node_size = node->size;

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
    if (required_units < node_size) {
      if (1 < allocated_units) {
        rjn_metadata_set(rjn, first_unit + pad_units + allocated_units - 1,
                         RJN_META_CONTINUATION);
      }
      rjn_metadata_set(rjn, first_unit + pad_units + allocated_units,
                       RJN_META_UNARY);
      memset(rjn_allocation_unit(rjn, first_unit + pad_units + allocated_units),
             0, sizeof(rjn_node));
      rjn_free_chunk(rjn, first_unit + pad_units + allocated_units,
                     node_size - pad_units - allocated_units);
    }

    rjn_metadata_set_size(rjn, first_unit + pad_units, allocated_units);

    rjn_validate(rjn);
    return (uint8_t *)node + pad_bytes;
  }

  rjn_unlock_size_class(sc);

  rjn_validate(rjn);
  return NULL;
}

void *rjn_alloc(rjn_allocator *rjn, size_t alignment, size_t size) {
  debug_printf("alloc %lu %lu\n", alignment, size);
  // Emulate the behavior of malloc(0), which returns a different, valid
  // pointer every time it is called.
  if (size == 0) {
    size = 1;
  }

  size_t min_units =
      (size + rjn->allocation_unit_size - 1) / rjn->allocation_unit_size;
  uint64_t num_sclasses = rjn_num_size_classes(rjn);

  // First search the larger size classes.  Any node in any of those size
  // classes should be sufficient to satisfy the request (except possibly due
  // to alignment constraints).
  for (unsigned int scidx = 1 + rjn_find_size_class_index(rjn, min_units);
       scidx < num_sclasses; scidx++) {
    void *ptr = rjn_alloc_from_size_class(rjn, scidx, alignment, size);
    if (ptr) {
      return ptr;
    }
  }
  // OK we're desperate.  Try groveling through the size class for this
  // particular size to see if there's anything there.
  return rjn_alloc_from_size_class(
      rjn, rjn_find_size_class_index(rjn, min_units), alignment, size);
}

/*
 * Init and miscellaneous functions
 */

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
  rjn->num_allocation_units = space / (allocation_unit_size + 1);
  while (space < rjn->num_allocation_units * allocation_unit_size +
                     (rjn->num_allocation_units + META_ENTRIES_PER_BYTE - 1) /
                         META_ENTRIES_PER_BYTE +
                     sizeof(size_class) * rjn_num_size_classes(rjn)) {
    rjn->num_allocation_units--;
  }

  rjn->metadata_offset = rjn->region_size - (rjn->num_allocation_units +
                                             META_ENTRIES_PER_BYTE - 1) /
                                                META_ENTRIES_PER_BYTE;
  rjn->size_classes_offset =
      rjn->metadata_offset - sizeof(size_class) * rjn_num_size_classes(rjn);

  memset(rjn_metadata_vector(rjn), RJN_META_CONTINUATION,
         (rjn->num_allocation_units + META_ENTRIES_PER_BYTE - 1) /
             META_ENTRIES_PER_BYTE);

  size_class *scs = rjn_size_classes(rjn);
  uint64_t num_scs = rjn_num_size_classes(rjn);
  for (uint64_t i = 0; i < num_scs; i++) {
    rjn_node *head = &scs[i].head;
    head->prev = 0;
    head->next = 0;
    head->size = 0;
    scs[i].num_chunks = 0;
    scs[i].num_units = 0;
  }

  rjn_node *first_node = rjn_allocation_unit(rjn, 0);
  rjn_node *last_node = rjn_allocation_unit(rjn, rjn->num_allocation_units - 1);
  first_node->size = rjn->num_allocation_units;
  last_node->size = rjn->num_allocation_units;
  rjn_metadata_set(rjn, 0, RJN_META_FREE);
  rjn_metadata_set(rjn, rjn->num_allocation_units - 1, RJN_META_FREE);
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