#include <stddef.h>

typedef struct rjn_allocator rjn_allocator;

// Setup an allocator in a region of memory.
int rjn_init(rjn_allocator *rjn, size_t region_size,
             size_t allocation_unit_size);
void rjn_deinit(rjn_allocator *rjn);

void *rjn_alloc(rjn_allocator *rjn, size_t alignment, size_t size);
void rjn_free(rjn_allocator *rjn, void *ptr);
void *rjn_realloc(rjn_allocator *rjn, void *ptr, size_t alignment,
                  size_t new_bytes);

size_t rjn_size(const rjn_allocator *rjn);
size_t rjn_allocation_unit_size(const rjn_allocator *rjn);
void *rjn_start(const rjn_allocator *rjn);
void *rjn_end(const rjn_allocator *rjn);
void rjn_print_allocation_stats(const rjn_allocator *rjn);