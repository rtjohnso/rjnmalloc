#include <stddef.h>

typedef void * (*alloc_func)(void *, size_t, size_t);
typedef void (*free_func)(void *, void *);
typedef void *(*realloc_func)(void *, void *, size_t, size_t);

typedef struct {
    alloc_func alloc;
    free_func free;
    realloc_func realloc;
} allocator_ops;

extern allocator_ops default_allocator_ops;
