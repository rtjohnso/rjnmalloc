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

struct rjn {
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
  uint64_t prev; // offset, in bytes, from start of rjn
  uint64_t next; // offset, in bytes, from start of rjn
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
uint64_t rjn_offset(const rjn *rj, const void *ptr) {
  assert((uint8_t *)ptr >= (uint8_t *)rj);
  assert((uint8_t *)ptr < (uint8_t *)rj + rj->region_size);
  return (uint8_t *)ptr - (uint8_t *)rj;
}

static void *rjn_pointer(const rjn *rj, uint64_t offset) {
  assert(offset < rj->region_size);
  return (void *)((uint8_t *)rj + offset);
}

static uint8_t *rjn_allocation_units(const rjn *rj) {
  return (uint8_t *)rjn_pointer(rj, rj->allocation_units_offset);
}

static rjn_node *rjn_allocation_unit(const rjn *rj, uint64_t i) {
  assert(i < rj->num_allocation_units);
  uint8_t *p = rjn_allocation_units(rj);
  return (rjn_node *)(p + i * rj->allocation_unit_size);
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

static uint8_t *rjn_metadata_vector(const rjn *rj) {
  return (uint8_t *)rjn_pointer(rj, rj->metadata_offset);
}

static uint8_t rjn_metadata_get(const rjn *rj, uint64_t unit) {
  assert(unit < rj->num_allocation_units);
  const uint8_t *metadata = rjn_metadata_vector(rj);
  return META_GET(metadata, unit);
}

static int rjn_metadata_cas(const rjn *rj, uint64_t unit, uint8_t old,
                            uint8_t new) {
  assert(unit < rj->num_allocation_units);
  uint8_t *metadata = rjn_metadata_vector(rj);
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

static void rjn_metadata_set(const rjn *rj, uint64_t unit, uint8_t value) {
  assert(unit < rj->num_allocation_units);
  uint8_t *metadata = rjn_metadata_vector(rj);
  debug_printf("metadata_set %lu %s %s\n", unit,
               META_GET(metadata, unit) < 4 ? metaname[META_GET(metadata, unit)]
                                            : "size?",
               metaname[value]);

  uint8_t mask = BYTE_INSERT(META_GET(metadata, unit) ^ value, unit);
  __sync_fetch_and_xor(&VECTOR_BYTE(metadata, unit), mask);
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
static int rjn_metadata_try_set_end_to_free(const rjn *rj, uint64_t unit) {
  assert(unit < rj->num_allocation_units);
  if (unit == rj->num_allocation_units - 1) {
    rjn_metadata_set(rj, unit, RJN_META_FREE);
    return 1;
  }

  uint8_t *metadata = rjn_metadata_vector(rj);
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
static int rjn_metadata_try_set_start_to_free(const rjn *rj, uint64_t unit) {
  assert(unit < rj->num_allocation_units);
  if (unit == 0) {
    rjn_metadata_set(rj, unit, RJN_META_FREE);
    return 1;
  }

  uint8_t *metadata = rjn_metadata_vector(rj);
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
      (unit == rj->num_allocation_units - 1 or
      metadata[uniit + 1] == RJN_META_UNARY or
      metadata[unit + 1] == RJN_META_BINARY)
 * then
     set metadata[unit] to RJN_META_FREE
     return 1
 * else
     return 0
 */
static int rjn_metadata_try_set_singleton_to_free(const rjn *rj,
                                                  uint64_t unit) {
  assert(unit < rj->num_allocation_units);
  if (unit == 0 && unit == rj->num_allocation_units - 1) {
    rjn_metadata_set(rj, unit, RJN_META_FREE);
    return 1;
  } else if (unit == 0) {
    return rjn_metadata_try_set_end_to_free(rj, unit);
  } else if (unit == rj->num_allocation_units - 1) {
    return rjn_metadata_try_set_start_to_free(rj, unit);
  } else {
  }

  uint8_t *metadata = rjn_metadata_vector(rj);
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

static void rjn_metadata_set_size(const rjn *rj, uint64_t first_unit,
                                  uint64_t units) {
  assert(first_unit + units <= rj->num_allocation_units);
  uint8_t *metadata = rjn_metadata_vector(rj);
  uint8_t fum = META_GET(metadata, first_unit);
  assert(fum == RJN_META_UNARY || fum == RJN_META_BINARY);
  if (units < RJN_MIN_BINARY_UNITS) {
    if (fum != RJN_META_UNARY) {
      rjn_metadata_set(rj, first_unit, RJN_META_UNARY);
    }
    assert(first_unit + units <= rj->num_allocation_units);
    uint64_t i;
    for (i = 1; i < units - 1 && (first_unit + i) % META_ENTRIES_PER_BYTE;
         i++) {
      rjn_metadata_set(rj, first_unit + i, RJN_META_CONTINUATION);
    }
    memset(&metadata[VECTOR_INDEX(first_unit + i)], META_CONTINUATION_BYTE,
           (units - i) / META_ENTRIES_PER_BYTE);
    for (i = 1; i < units - 1; i++) {
      rjn_metadata_set(rj, first_unit + i, RJN_META_CONTINUATION);
    }
  } else {
    uint64_t size_start =
        (first_unit + META_ENTRIES_PER_BYTE) / META_ENTRIES_PER_BYTE;
    assert(size_start + sizeof(uint64_t) <=
           (first_unit + units) / META_ENTRIES_PER_BYTE);
    if (fum != RJN_META_BINARY) {
      rjn_metadata_set(rj, first_unit, RJN_META_BINARY);
    }
    memcpy(&metadata[size_start], &units, sizeof(uint64_t));
  }
}

static uint64_t rjn_metadata_get_size(const rjn *rj, uint64_t first_unit) {
  assert(first_unit < rj->num_allocation_units);
  uint8_t *metadata = rjn_metadata_vector(rj);
  uint64_t size;
  if (META_GET(metadata, first_unit) == RJN_META_UNARY) {
    size = 1;
    while (first_unit + size < rj->num_allocation_units &&
           (first_unit + size) % META_ENTRIES_PER_BYTE &&
           META_GET(metadata, first_unit + size) == RJN_META_CONTINUATION) {
      size++;
    }
    while (
        first_unit + size + META_ENTRIES_PER_BYTE <= rj->num_allocation_units &&
        metadata[VECTOR_INDEX(first_unit + size)] == META_CONTINUATION_BYTE) {
      size += META_ENTRIES_PER_BYTE;
    }
    while (first_unit + size < rj->num_allocation_units &&
           META_GET(metadata, first_unit + size) == RJN_META_CONTINUATION) {
      size++;
    }
  } else {
    assert(first_unit + RJN_MIN_BINARY_UNITS <= rj->num_allocation_units);
    memcpy(
        &size,
        &metadata[(first_unit + META_ENTRIES_PER_BYTE) / META_ENTRIES_PER_BYTE],
        sizeof(uint64_t));
  }
  return size;
}

static void rjn_metadata_erase_size(const rjn *rj, uint64_t first_unit,
                                    uint64_t units) {
  assert(first_unit + units <= rj->num_allocation_units);
  uint8_t *metadata = rjn_metadata_vector(rj);
  if (units < RJN_MIN_BINARY_UNITS) {
    assert(META_GET(metadata, first_unit) == RJN_META_UNARY);
    uint64_t i;
    for (i = 1; i < units - 1 && (first_unit + i) % META_ENTRIES_PER_BYTE;
         i++) {
      rjn_metadata_set(rj, first_unit + i, RJN_META_CONTINUATION);
    }
    while (i + META_ENTRIES_PER_BYTE <= units - 1) {
      metadata[(first_unit + i) / META_ENTRIES_PER_BYTE] =
          META_CONTINUATION_BYTE;
      i += META_ENTRIES_PER_BYTE;
    }
    while (i < units - 1) {
      rjn_metadata_set(rj, first_unit + i, RJN_META_CONTINUATION);
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
debug_code static void node_to_string(const rjn *rj, const rjn_node *node,
                                      char *buffer, size_t buffer_size) {
  uint64_t offset = rjn_offset(rj, node);
  assert((offset - rj->allocation_units_offset) % rj->allocation_unit_size ==
         0);
  uint64_t aunum =
      (offset - rj->allocation_units_offset) / rj->allocation_unit_size;
  assert(aunum < rj->num_allocation_units);

  snprintf(buffer, buffer_size,
           "    node %p\n"
           "    offset %lu\n"
           "    aunum %lu\n"
           "    prev %lu\n"
           "    next %lu\n"
           "    size %lu\n"
           "    %s %s",
           (void *)node, offset, aunum, node->prev, node->next, node->size,
           metaname[rjn_metadata_get(rj, aunum)],
           0 < node->size
               ? metaname[rjn_metadata_get(rj, aunum + node->size - 1)]
               : "-");
}

#define node_string(rj, node)                                                  \
  (({                                                                          \
     struct {                                                                  \
       char buffer[256];                                                       \
     } b;                                                                      \
     node_to_string((rj), (node), b.buffer, sizeof(b.buffer));                 \
     b;                                                                        \
   }).buffer)

#define offset_string(rj, off)                                                 \
  (({                                                                          \
     struct {                                                                  \
       char buffer[256];                                                       \
     } b;                                                                      \
     if (off) {                                                                \
       node_to_string((rj), rjn_pointer(rj, off), b.buffer, sizeof(b.buffer)); \
     } else {                                                                  \
       snprintf(b.buffer, sizeof(b.buffer), "    (null)");                     \
     }                                                                         \
     b;                                                                        \
   }).buffer)

#define aunum_string(rj, aunum)                                                \
  (({                                                                          \
     struct {                                                                  \
       char buffer[256];                                                       \
     } b;                                                                      \
     node_to_string((rj), rjn_allocation_unit(rj, aunum), b.buffer,            \
                    sizeof(b.buffer));                                         \
     b;                                                                        \
   }).buffer)

/*
 * Size-class operations
 */

// Size class i contains chunks of size in the range [2^i, 2^(i+1)) allocation
// units.
static uint64_t rjn_num_size_classes(const rjn *rj) {
  return 64 - _lzcnt_u64(rj->num_allocation_units);
}

static size_class *rjn_size_classes(const rjn *rj) {
  return (size_class *)rjn_pointer(rj, rj->size_classes_offset);
}

/* Return the size class where a free chunk of size 'units' allocation units
   should be stored. */
static unsigned int rjn_find_size_class_index(const rjn *rj, size_t units) {
  uint64_t num_sclasses = rjn_num_size_classes(rj);

  assert(units);
  unsigned int fl = 63 - _lzcnt_u64(units);
  if (num_sclasses <= fl) {
    return num_sclasses;
  } else {
    return fl;
  }
}

static size_class *rjn_find_size_class(const rjn *rj, size_t units) {
  unsigned int scidx = rjn_find_size_class_index(rj, units);
  if (scidx < rjn_num_size_classes(rj)) {
    return &rjn_size_classes(rj)[scidx];
  } else {
    return &rjn_size_classes(rj)[scidx - 1];
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

static void rjn_prepend(const rjn *rj, size_class *sclass, rjn_node *node) {
  assert((void *)node != (void *)rj);
  assert(node->next == 0);
  assert(node->prev == 0);
  node->next = sclass->head.next;
  node->prev = rjn_offset(rj, sclass);
  if (sclass->head.next) {
    rjn_node *next = rjn_pointer(rj, sclass->head.next);
    assert(next != node);
    assert(next->prev == rjn_offset(rj, sclass));
    next->prev = rjn_offset(rj, node);
  }
  sclass->head.next = rjn_offset(rj, node);
  sclass->num_chunks++;
  sclass->num_units += node->size;
}

static void rjn_remove(const rjn *rj, size_class *sclass, rjn_node *node) {
  assert((void *)node != (void *)rj);
  assert(node->prev != 0);
  rjn_node *prev = rjn_pointer(rj, node->prev);
  assert(prev->next == rjn_offset(rj, node));
  prev->next = node->next;
  if (node->next) {
    rjn_node *next = rjn_pointer(rj, node->next);
    assert(next->prev == rjn_offset(rj, node));
    next->prev = node->prev;
  }
  node->next = node->prev = 0;
  sclass->num_chunks--;
  sclass->num_units -= node->size;
}

/*
 * Debugging validation code
 */

typedef void (*chunk_walk_func)(rjn *rj, uint64_t start_unit, uint64_t size,
                                void *arg);

static void rjn_walk_chunks(rjn *rj, chunk_walk_func func, void *arg) {
  uint64_t curr_unit = 0;
  while (curr_unit < rj->num_allocation_units) {
    uint64_t size = 0;
    switch (rjn_metadata_get(rj, curr_unit)) {
    case RJN_META_FREE: {
      rjn_node *node = rjn_allocation_unit(rj, curr_unit);
      size = node->size;
    } break;
    case RJN_META_UNARY:
    case RJN_META_BINARY:
      size = rjn_metadata_get_size(rj, curr_unit);
      break;
    default:
      assert(0);
      break;
    }
    assert(size);
    if (1 < RJN_DEBUG) {
      for (uint64_t i = rjn_metadata_get(rj, curr_unit) == RJN_META_BINARY
                            ? 1 + sizeof(uint64_t)
                            : 1;
           i < size - 1; i++) {
        assert(rjn_metadata_get(rj, curr_unit + i) == RJN_META_CONTINUATION);
      }
    }
    if (1 < size) {
      if (rjn_metadata_get(rj, curr_unit) == RJN_META_FREE) {
        assert(rjn_metadata_get(rj, curr_unit + size - 1) == RJN_META_FREE);
      } else {
        assert(rjn_metadata_get(rj, curr_unit + size - 1) ==
               RJN_META_CONTINUATION);
      }
    }
    func(rj, curr_unit, size, arg);
    curr_unit += size;
    assert(curr_unit <= rj->num_allocation_units);
  }
}

static void rjn_count_chunks(rjn *rj, uint64_t start_unit, uint64_t size,
                             void *arg) {
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

static uint64_t rjn_find_chunk_in_array(const chunk_array *ca, uint64_t start) {
  chunk key = {.start = start};
  void *p = bsearch(&key, ca->chunks, ca->num_chunks, sizeof(chunk),
                    compare_chunk_start);
  assert(p);
  uint64_t i = (chunk *)p - ca->chunks;
  assert(i < ca->num_chunks);
  assert(ca->chunks[i].start == start);
  return i;
}

static void rjn_collect_chunks(rjn *rj, uint64_t start_unit, uint64_t size,
                               void *arg) {
  chunk_array *ca = (chunk_array *)arg;
  assert(ca->num_chunks < ca->max_chunks);
  ca->chunks[ca->num_chunks].start = start_unit;
  ca->chunks[ca->num_chunks].units = size;
  ca->chunks[ca->num_chunks].sc = NULL;
  ca->num_chunks++;
}

static void rjn_validate_size_classes(const rjn *rj, chunk_array *ca) {
  uint64_t num_sclasses = rjn_num_size_classes(rj);
  size_class *scs = rjn_size_classes(rj);
  for (uint64_t i = 0; i < num_sclasses; i++) {
    size_class *sc = &scs[i];
    for (uint64_t curr_off = sc->head.next; curr_off;
         curr_off = ((rjn_node *)rjn_pointer(rj, curr_off))->next) {
      uint64_t start_unit =
          (curr_off - rj->allocation_units_offset) / rj->allocation_unit_size;
      uint64_t c = rjn_find_chunk_in_array(ca, start_unit);
      assert(ca->chunks[c].sc == NULL);
      ca->chunks[c].sc = sc;
    }
  }

  for (uint64_t i = 0; i < ca->num_chunks; i++) {
    if (rjn_metadata_get(rj, ca->chunks[i].start) == RJN_META_FREE) {
      assert(ca->chunks[i].sc);
    }
  }
}

debug_code static void rjn_validate(rjn *rj) {
  if (RJN_DEBUG) {
    uint64_t num_chunks = 0;
    rjn_walk_chunks(rj, rjn_count_chunks, &num_chunks);
    chunk_array *ca = malloc(sizeof(chunk_array) + num_chunks * sizeof(chunk));
    ca->num_chunks = 0;
    ca->max_chunks = num_chunks;
    rjn_walk_chunks(rj, rjn_collect_chunks, ca);
    rjn_validate_size_classes(rj, ca);
    if (RJN_DUMP_CHUNKS) {
      printf("chunks\n");
      for (uint64_t i = 0; i < ca->num_chunks; i++) {
        printf("  chunk %12lu %12lu %s %s\n", ca->chunks[i].start,
               ca->chunks[i].units,
               metaname[rjn_metadata_get(rj, ca->chunks[i].start)],
               1 < ca->chunks[i].units
                   ? metaname[rjn_metadata_get(rj, ca->chunks[i].start +
                                                       ca->chunks[i].units - 1)]
                   : "");
      }
    }
    free(ca);
  }
}

/*
 * Chunk freeing functions
 */

static uint64_t rjn_grab_predecessor_if_free(const rjn *rj, uint64_t my_start) {
  if (my_start == 0) {
    return 0;
  }
  uint64_t pred_last = my_start - 1;
  if (!rjn_metadata_cas(rj, pred_last, RJN_META_FREE, RJN_META_UNARY)) {
    return my_start;
  }
  rjn_node *pred_node = rjn_allocation_unit(rj, pred_last);
  assert(pred_node->size <= my_start);
  uint64_t pred_start = my_start - pred_node->size;
  if (pred_start < pred_last) {
    if (!rjn_metadata_cas(rj, pred_start, RJN_META_FREE, RJN_META_UNARY)) {
      rjn_metadata_set(rj, pred_last, RJN_META_FREE);
      return my_start;
    }
    rjn_metadata_set(rj, pred_last, RJN_META_CONTINUATION);
  }
  return pred_start;
}

static uint64_t rjn_grab_successor_if_free(const rjn *rj, uint64_t my_start,
                                           uint64_t my_units) {
  uint64_t succ_start = my_start + my_units;
  if (rj->num_allocation_units <= succ_start) {
    return 0;
  }
  if (!rjn_metadata_cas(rj, succ_start, RJN_META_FREE, RJN_META_UNARY)) {
    return 0;
  }
  rjn_node *succ_node = rjn_allocation_unit(rj, succ_start);
  uint64_t succ_last = succ_start + succ_node->size - 1;
  if (succ_start < succ_last) {
    while (!rjn_metadata_cas(rj, succ_last, RJN_META_FREE,
                             RJN_META_CONTINUATION)) {
      _mm_pause();
    }
  }
  return succ_start;
}

static void rjn_free_chunk(const rjn *rj, uint64_t start_unit, uint64_t units) {
  rjn_node *start_node = NULL;

restart:
  debug_printf("free_chunk %lu %lu\n", start_unit, units);

  assert(rjn_metadata_get(rj, start_unit) == RJN_META_UNARY ||
         rjn_metadata_get(rj, start_unit) == RJN_META_BINARY);
  if (1 < units) {
    assert(rjn_metadata_get(rj, start_unit + units - 1) ==
           RJN_META_CONTINUATION);
  }

  rjn_metadata_erase_size(rj, start_unit, units);

  start_node = rjn_allocation_unit(rj, start_unit);
  rjn_node *last_node = rjn_allocation_unit(rj, start_unit + units - 1);

  start_node->size = units;
  last_node->size = units;

  size_class *sc = rjn_find_size_class(rj, units);
  rjn_lock_size_class(sc);
  rjn_prepend(rj, sc, start_node);
  rjn_unlock_size_class(sc);

  if (units == 1) {

    if (!rjn_metadata_try_set_singleton_to_free(rj, start_unit)) {
      rjn_lock_size_class(sc);
      rjn_remove(rj, sc, start_node);
      rjn_unlock_size_class(sc);

      // Try to merge with our predecessor.
      uint64_t new_start = rjn_grab_predecessor_if_free(rj, start_unit);
      if (new_start < start_unit) {
        rjn_node *pred_start_node = rjn_allocation_unit(rj, new_start);
        debug_printf("free_chunk merging %lu %lu with %lu %lu\n", start_unit,
                     units, new_start, pred_start_node->size);
        size_class *pred_sc = rjn_find_size_class(rj, pred_start_node->size);
        rjn_lock_size_class(pred_sc);
        rjn_remove(rj, pred_sc, pred_start_node);
        rjn_unlock_size_class(pred_sc);
        rjn_metadata_set(rj, start_unit, RJN_META_CONTINUATION);
        start_unit = new_start;
        units += pred_start_node->size;
      }

      // Try to merge with our successor.
      uint64_t successor_start =
          rjn_grab_successor_if_free(rj, start_unit, units);
      if (successor_start) {
        rjn_node *succ_start_node = rjn_allocation_unit(rj, successor_start);
        debug_printf("free_chunk merging %lu %lu with %lu %lu\n", start_unit,
                     units, successor_start, succ_start_node->size);
        size_class *succ_sc = rjn_find_size_class(rj, succ_start_node->size);
        rjn_lock_size_class(succ_sc);
        rjn_remove(rj, succ_sc, succ_start_node);
        rjn_unlock_size_class(succ_sc);
        rjn_metadata_set(rj, successor_start, RJN_META_CONTINUATION);
        units += succ_start_node->size;
      }

      goto restart;
    }

  } else {

    if (!rjn_metadata_try_set_start_to_free(rj, start_unit)) {
      rjn_lock_size_class(sc);
      rjn_remove(rj, sc, start_node);
      rjn_unlock_size_class(sc);

      // Try to merge with our predecessor.
      uint64_t new_start = rjn_grab_predecessor_if_free(rj, start_unit);
      if (new_start < start_unit) {
        rjn_node *pred_start_node = rjn_allocation_unit(rj, new_start);
        debug_printf("free_chunk merging %lu %lu with %lu %lu\n", start_unit,
                     units, new_start, pred_start_node->size);
        size_class *pred_sc = rjn_find_size_class(rj, pred_start_node->size);
        rjn_lock_size_class(pred_sc);
        rjn_remove(rj, pred_sc, pred_start_node);
        rjn_unlock_size_class(pred_sc);
        rjn_metadata_set(rj, start_unit, RJN_META_CONTINUATION);
        start_unit = new_start;
        units += pred_start_node->size;
      }

      goto restart;
    }

    if (!rjn_metadata_try_set_end_to_free(rj, start_unit + units - 1)) {
      if (!rjn_metadata_cas(rj, start_unit, RJN_META_FREE, RJN_META_UNARY)) {
        // Someone else is already trying to allocate this chunk or to merge
        // with this chunk from the left, so we can finish freeing the chunk and
        // return.
        rjn_metadata_set(rj, start_unit + units - 1, RJN_META_FREE);
      } else {
        rjn_lock_size_class(sc);
        rjn_remove(rj, sc, start_node);
        rjn_unlock_size_class(sc);

        // Try to merge with our successor.
        uint64_t successor_start =
            rjn_grab_successor_if_free(rj, start_unit, units);
        if (successor_start) {
          rjn_node *succ_start_node = rjn_allocation_unit(rj, successor_start);
          debug_printf("free_chunk merging %lu %lu with %lu %lu\n", start_unit,
                       units, successor_start, succ_start_node->size);
          size_class *succ_sc = rjn_find_size_class(rj, succ_start_node->size);
          rjn_lock_size_class(succ_sc);
          rjn_remove(rj, succ_sc, succ_start_node);
          rjn_unlock_size_class(succ_sc);
          rjn_metadata_set(rj, successor_start, RJN_META_CONTINUATION);
          units += succ_start_node->size;
        }

        goto restart;
      }
    }
  }
}

void rjn_free(rjn *rj, void *ptr) {
  rjn_validate(rj);
  uint64_t first_unit = (rjn_offset(rj, ptr) - rj->allocation_units_offset) /
                        rj->allocation_unit_size;
  uint64_t size = rjn_metadata_get_size(rj, first_unit);
  assert(size);
  assert(size <= rj->num_allocation_units);
  assert(rjn_metadata_get(rj, first_unit) == RJN_META_UNARY ||
         rjn_metadata_get(rj, first_unit) == RJN_META_BINARY);
  assert(size == 1 ||
         rjn_metadata_get(rj, first_unit + size - 1) == RJN_META_CONTINUATION);
  rjn_node *node = rjn_allocation_unit(rj, first_unit);
  node->prev = node->next = 0;
  rjn_free_chunk(rj, first_unit, size);
  rjn_validate(rj);
}

/*
 * Allocation functions
 */

static void *rjn_alloc_from_size_class(rjn *rj, unsigned int scidx,
                                       size_t alignment_bytes, size_t bytes) {
  size_class *sc = &rjn_size_classes(rj)[scidx];
  if (!sc->num_chunks) {
    return NULL;
  }

restart:
  rjn_validate(rj);

  rjn_lock_size_class(sc);

  uint64_t next_off;
  for (uint64_t curr_off = sc->head.next; curr_off; curr_off = next_off) {
    next_off = ((rjn_node *)rjn_pointer(rj, curr_off))->next;

    uint64_t first_unit =
        (curr_off - rj->allocation_units_offset) / rj->allocation_unit_size;

    assert(rjn_metadata_get(rj, first_unit) != RJN_META_CONTINUATION);
    if (!rjn_metadata_cas(rj, first_unit, RJN_META_FREE, RJN_META_UNARY)) {
      // If the CAS fails, that means that someone else is freeing the
      // previous chunk and merging their free chunk with the current one.
      continue;
    }
    rjn_node *node = rjn_pointer(rj, curr_off);

    // Check if the current chunk is large enough to satisfy the request,
    // including alignment padding.
    uint64_t pad_bytes = 0;
    uint64_t pad_units = 0;
    if (1 < alignment_bytes) {
      pad_bytes = alignment_bytes - ((uint64_t)node % alignment_bytes);
      pad_units = pad_bytes / rj->allocation_unit_size;
    }
    uint64_t required_bytes = pad_bytes + bytes;
    uint64_t required_units = (required_bytes + rj->allocation_unit_size - 1) /
                              rj->allocation_unit_size;
    uint64_t allocated_units = required_units - pad_units;

    if (node->size < required_units) {
      if (rjn_metadata_try_set_start_to_free(rj, first_unit)) {
        continue;
      }
      // Aw crud.  Someone freed the preceding chunk, so we can't just quietly
      // put this one back.  We have to merge it with the preceding chunk.
      // Merging chunks can require interacting with lots of different size
      // classes (because merges can cascade), so we're just gonna give up our
      // lock on this size class and restart from scratch.
      if (1 < node->size) {
        while (!rjn_metadata_cas(rj, first_unit + node->size - 1, RJN_META_FREE,
                                 RJN_META_CONTINUATION)) {
          _mm_pause();
        }
      }
      rjn_remove(rj, sc, node);
      rjn_unlock_size_class(sc);
      rjn_free_chunk(rj, first_unit, node->size);
      goto restart;
    }

    // Finish allocating the chunk.
    if (1 < node->size) {
      while (!rjn_metadata_cas(rj, first_unit + node->size - 1, RJN_META_FREE,
                               RJN_META_CONTINUATION)) {
        _mm_pause();
      }
    }
    rjn_remove(rj, sc, node);
    rjn_unlock_size_class(sc);
    uint64_t node_size = node->size;

    // Give back any alignment pad at the beginning
    if (pad_units) {
      if (1 < pad_units) {
        rjn_metadata_set(rj, first_unit + pad_units - 1, RJN_META_CONTINUATION);
      }
      rjn_metadata_set(rj, first_unit + pad_units, RJN_META_UNARY);
      rjn_free_chunk(rj, first_unit, pad_units);
    }

    // Give back any remaining space at the end
    if (required_units < node_size) {
      if (1 < allocated_units) {
        rjn_metadata_set(rj, first_unit + pad_units + allocated_units - 1,
                         RJN_META_CONTINUATION);
      }
      rjn_metadata_set(rj, first_unit + pad_units + allocated_units,
                       RJN_META_UNARY);
      memset(rjn_allocation_unit(rj, first_unit + pad_units + allocated_units),
             0, sizeof(rjn_node));
      rjn_free_chunk(rj, first_unit + pad_units + allocated_units,
                     node_size - pad_units - allocated_units);
    }

    rjn_metadata_set_size(rj, first_unit + pad_units, allocated_units);

    rjn_validate(rj);
    return (uint8_t *)node + pad_bytes;
  }

  rjn_unlock_size_class(sc);

  rjn_validate(rj);
  return NULL;
}

void *rjn_alloc(rjn *rj, size_t alignment, size_t size) {
  debug_printf("alloc %lu %lu\n", alignment, size);
  // Emulate the behavior of malloc(0), which returns a different, valid
  // pointer every time it is called.
  if (size == 0) {
    size = 1;
  }

  size_t min_units =
      (size + rj->allocation_unit_size - 1) / rj->allocation_unit_size;
  uint64_t num_sclasses = rjn_num_size_classes(rj);

  // First search the larger size classes.  Any node in any of those size
  // classes should be sufficient to satisfy the request (except possibly due
  // to alignment constraints).
  for (unsigned int scidx = 1 + rjn_find_size_class_index(rj, min_units);
       scidx < num_sclasses; scidx++) {
    void *ptr = rjn_alloc_from_size_class(rj, scidx, alignment, size);
    if (ptr) {
      return ptr;
    }
  }
  // OK we're desperate.  Try groveling through the size class for this
  // particular size to see if there's anything there.
  return rjn_alloc_from_size_class(rj, rjn_find_size_class_index(rj, min_units),
                                   alignment, size);
}

void *rjn_realloc(rjn *rj, void *ptr, size_t alignment, size_t new_bytes) {
  debug_printf("realloc %p %lu %lu\n", ptr, alignment, new_bytes);
  rjn_validate(rj);

  if (ptr == NULL) {
    return rjn_alloc(rj, alignment, new_bytes);
  }
  if (new_bytes == 0) {
    rjn_free(rj, ptr);
    return NULL;
  }
  uint64_t first_unit = (rjn_offset(rj, ptr) - rj->allocation_units_offset) /
                        rj->allocation_unit_size;
  uint64_t old_units = rjn_metadata_get_size(rj, first_unit);
  assert(old_units != 0);
  uint64_t old_bytes = old_units * rj->allocation_unit_size -
                       ((uint64_t)ptr % rj->allocation_unit_size);

  size_t copy_size = old_bytes < new_bytes ? old_bytes : new_bytes;

  if ((uint64_t)ptr % alignment) {
    void *new_ptr = rjn_alloc(rj, alignment, new_bytes);
    if (new_ptr) {
      memcpy(new_ptr, ptr, copy_size);
      rjn_free(rj, ptr);
    }
    return new_ptr;
  }

  uint64_t required_units = (((uint64_t)ptr % rj->allocation_unit_size) +
                             new_bytes + rj->allocation_unit_size - 1) /
                            rj->allocation_unit_size;

  if (required_units == old_units) {
    return ptr;
  }

  if (new_bytes <= old_bytes) {
    rjn_metadata_erase_size(rj, first_unit, old_units);
    if (1 < required_units) {
      rjn_metadata_set(rj, first_unit + required_units - 1,
                       RJN_META_CONTINUATION);
    }
    rjn_metadata_set(rj, first_unit + required_units, RJN_META_UNARY);
    memset(rjn_allocation_unit(rj, first_unit + required_units), 0,
           sizeof(rjn_node));
    rjn_free_chunk(rj, first_unit + required_units, old_units - required_units);
    rjn_metadata_set_size(rj, first_unit, required_units);
    return ptr;
  } else {
    uint64_t succ_start = rjn_grab_successor_if_free(rj, first_unit, old_units);
    if (first_unit < succ_start) {
      rjn_node *succ_start_node = rjn_allocation_unit(rj, succ_start);
      size_class *sc = rjn_find_size_class(rj, succ_start_node->size);
      if (new_bytes <=
          old_bytes + succ_start_node->size * rj->allocation_unit_size) {
        rjn_lock_size_class(sc);
        rjn_remove(rj, sc, succ_start_node);
        rjn_unlock_size_class(sc);
        rjn_metadata_set(rj, succ_start, RJN_META_CONTINUATION);
        rjn_metadata_erase_size(rj, first_unit, old_units);
        if (required_units < old_units + succ_start_node->size) {
          assert(1 < required_units);
          rjn_metadata_set(rj, first_unit + required_units - 1,
                           RJN_META_CONTINUATION);
          rjn_metadata_set(rj, first_unit + required_units, RJN_META_UNARY);
          memset(rjn_allocation_unit(rj, first_unit + required_units), 0,
                 sizeof(rjn_node));
          rjn_free_chunk(rj, first_unit + required_units,
                         old_units + succ_start_node->size - required_units);
        }
        rjn_metadata_set_size(rj, first_unit, required_units);
        return ptr;
      } else {
        rjn_lock_size_class(sc);
        rjn_remove(rj, sc, succ_start_node);
        rjn_unlock_size_class(sc);
        rjn_free_chunk(rj, succ_start, succ_start_node->size);
      }
    }

    void *new_ptr = rjn_alloc(rj, alignment, new_bytes);
    if (new_ptr) {
      memcpy(new_ptr, ptr, copy_size);
      rjn_free(rj, ptr);
    }
    return new_ptr;
  }
}

/*
 * Init and miscellaneous functions
 */

int rjn_init(rjn *rj, size_t region_size, size_t allocation_unit_size) {
  if (region_size < sizeof(rj)) {
    return -1;
  }
  if (allocation_unit_size < MIN_ALLOCATION_UNIT_SIZE) {
    return -1;
  }

  rj->region_size = region_size;
  rj->allocation_unit_size = allocation_unit_size;

  rj->allocation_units_offset = sizeof(rjn);
  uint64_t alignment =
      ((uint64_t)rj + rj->allocation_units_offset) % allocation_unit_size;
  if (alignment) {
    rj->allocation_units_offset += allocation_unit_size - alignment;
  }
  if (rj->region_size < rj->allocation_units_offset) {
    return -1;
  }

  uint64_t space = rj->region_size - rj->allocation_units_offset;
  rj->num_allocation_units = space / (allocation_unit_size + 1);
  while (space < rj->num_allocation_units * allocation_unit_size +
                     (rj->num_allocation_units + META_ENTRIES_PER_BYTE - 1) /
                         META_ENTRIES_PER_BYTE +
                     sizeof(size_class) * rjn_num_size_classes(rj)) {
    rj->num_allocation_units--;
  }

  rj->metadata_offset =
      rj->region_size - (rj->num_allocation_units + META_ENTRIES_PER_BYTE - 1) /
                            META_ENTRIES_PER_BYTE;
  rj->size_classes_offset =
      rj->metadata_offset - sizeof(size_class) * rjn_num_size_classes(rj);

  memset(rjn_metadata_vector(rj), RJN_META_CONTINUATION,
         (rj->num_allocation_units + META_ENTRIES_PER_BYTE - 1) /
             META_ENTRIES_PER_BYTE);

  size_class *scs = rjn_size_classes(rj);
  uint64_t num_scs = rjn_num_size_classes(rj);
  for (uint64_t i = 0; i < num_scs; i++) {
    rjn_node *head = &scs[i].head;
    head->prev = 0;
    head->next = 0;
    head->size = 0;
    scs[i].num_chunks = 0;
    scs[i].num_units = 0;
  }

  rjn_node *first_node = rjn_allocation_unit(rj, 0);
  rjn_node *last_node = rjn_allocation_unit(rj, rj->num_allocation_units - 1);
  first_node->size = rj->num_allocation_units;
  last_node->size = rj->num_allocation_units;
  rjn_metadata_set(rj, 0, RJN_META_FREE);
  rjn_metadata_set(rj, rj->num_allocation_units - 1, RJN_META_FREE);
  rjn_prepend(rj, &scs[num_scs - 1], first_node);

  return 0;
}

void rjn_deinit(rjn *rj) {
  // Nothing to do.
}

size_t rjn_size(const rjn *rj) {
  return rj->num_allocation_units * rj->allocation_unit_size;
}

size_t rjn_allocation_unit_size(const rjn *rj) {
  return rj->allocation_unit_size;
}

void *rjn_start(const rjn *rj) {
  return rjn_pointer(rj, rj->allocation_units_offset);
}

void *rjn_end(const rjn *rj) {
  return rjn_pointer(rj,
                     rj->allocation_units_offset +
                         rj->num_allocation_units * rj->allocation_unit_size);
}

void rjn_print_allocation_stats(const rjn *rj) {
  uint64_t num_sclasses = rjn_num_size_classes(rj);
  size_class *scs = rjn_size_classes(rj);

  uint64_t total_free_units = 0;
  uint64_t total_free_chunks = 0;
  printf("----------------------------------------\n");
  for (uint64_t i = 0; i < num_sclasses; i++) {
    size_class *sc = &scs[i];
    total_free_units += sc->num_units;
    total_free_chunks += sc->num_chunks;
    printf("size class %2" PRIu64 ": %12" PRIu64 " chunks, %12" PRIu64
           " units\n",
           i, sc->num_chunks, sc->num_units);
  }
  printf("total free chunks: %" PRIu64 "\n", total_free_chunks);
  printf("total free units: %" PRIu64 "\n", total_free_units);
  printf("total free bytes: %" PRIu64 "\n",
         total_free_units * rj->allocation_unit_size);
}

allocator_ops rjn_allocator_ops = {
    .alloc = (alloc_func)rjn_alloc,
    .free = (free_func)rjn_free,
    .realloc = (realloc_func)rjn_realloc,
};