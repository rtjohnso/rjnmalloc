#include "allocator.h"
#include <stdlib.h>

void *default_realloc(void *ptr, size_t alignment, size_t size) {
  return realloc(ptr, size);
}

allocator default_allocator = {aligned_alloc, free, default_realloc};
