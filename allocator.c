#include "allocator.h"
#include <stdlib.h>

void *default_alloc(void *state, size_t alignment, size_t size) {
  return aligned_alloc(alignment, size);
}

void default_free(void *state, void *ptr) { free(ptr); }

void *default_realloc(void *state, void *ptr, size_t alignment, size_t size) {
  return realloc(ptr, size);
}

allocator_ops default_allocator_ops = {default_alloc, default_free,
                                       default_realloc};
