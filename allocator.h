#include <stddef.h>

typedef void * (*alloc_func)(size_t, size_t);
typedef void (*free_func)(void *);
typedef void *(*realloc_func)(void *, size_t, size_t);

typedef struct {
    alloc_func alloc;
    free_func free;
    realloc_func realloc;
} allocator;

extern allocator default_allocator;
