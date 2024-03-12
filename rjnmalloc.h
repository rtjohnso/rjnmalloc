#include <stddef.h>

typedef struct rjn_allocator rjn_allocator;

// Setup an allocator in a region of memory.
// When track allocations is set, the allocator will keep track of the
// amount of space that is allocated.  Note that this may cause contention in
// multithreaded environments.
int rjn_init(rjn_allocator *rjn, size_t region_size,
             size_t allocation_unit_size, int track_allocations);
void rjn_deinit(rjn_allocator *rjn);

void *rjn_alloc(rjn_allocator *rjn, size_t alignment, size_t size);
void rjn_free(rjn_allocator *rjn, void *ptr);

size_t rjn_size(rjn_allocator *rjn);
void *rjn_start(rjn_allocator *rjn);
void *rjn_end(rjn_allocator *rjn);
size_t rjn_allocated(rjn_allocator *rjn);