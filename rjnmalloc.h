#include "allocator.h"
#include <stddef.h>

typedef struct rjn rjn;

// Setup an allocator in a region of memory.
int rjn_init(rjn *rj, size_t region_size, size_t allocation_unit_size);
void rjn_deinit(rjn *rj);

void *rjn_alloc(rjn *rj, size_t alignment, size_t size);
void rjn_free(rjn *rj, void *ptr);
void *rjn_realloc(rjn *rj, void *ptr, size_t alignment, size_t new_bytes);

size_t rjn_size(const rjn *rj);
size_t rjn_allocation_unit_size(const rjn *rj);
void *rjn_start(const rjn *rj);
void *rjn_end(const rjn *rj);
void rjn_print_allocation_stats(const rjn *rj);

extern allocator_ops rjn_allocator_ops;
