#include "rjnmalloc.h"
#define _XOPEN_SOURCE
#include <assert.h>
#include <inttypes.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct allocation {
  uint8_t *p;
  size_t size;
  int c;
} allocation;

#define NALLOCATIONS (1000)

uint64_t nrounds = 1 << 20;

void malloc_test(rjn_allocator *hdr) {
  allocation allocations[NALLOCATIONS];

  memset(allocations, 0, sizeof(allocations));

  double lsize = log(rjn_size(hdr));
  uint8_t *rstart = (uint8_t *)rjn_start(hdr);
  uint8_t *rend = (uint8_t *)rjn_end(hdr);

  int nallocations = 0;

  for (uint64_t i = 0; i < nrounds; i++) {
    int i = rand() % NALLOCATIONS;
    if (allocations[i].p) {
      for (unsigned int j = 0; j < allocations[i].size; j++) {
        assert(allocations[i].p[j] == allocations[i].c);
      }
      printf("freeing %p\n", allocations[i].p);
      rjn_free(hdr, allocations[i].p);
      allocations[i].p = NULL;
      nallocations--;
    } else {
      allocations[i].size = exp(lsize * drand48());
      allocations[i].p = rjn_alloc(hdr, 0, allocations[i].size);
      allocations[i].c = rand() % 256;
      if (allocations[i].p) {
        assert(rstart <= allocations[i].p);
        assert(allocations[i].p + allocations[i].size <= rend);
        memset(allocations[i].p, allocations[i].c, allocations[i].size);
        nallocations++;
      }
    }
    printf("nallocations: %d\n", nallocations);
  }

  for (int i = 0; i < NALLOCATIONS; i++) {
    if (allocations[i].p) {
      for (unsigned int j = 0; j < allocations[i].size; j++) {
        assert(allocations[i].p[j] == allocations[i].c);
      }
      printf("freeing %p\n", allocations[i].p);
      rjn_free(hdr, allocations[i].p);
      allocations[i].p = NULL;
      nallocations--;
    }
  }
  printf("nallocations: %d\n", nallocations);
}

void *malloc_test_thread(void *p) {
  malloc_test((rjn_allocator *)p);
  return NULL;
}

char rjnbuf[1 << 30];

#define NTHREADS (4)

int main(int argc, char **argv) {
  if (1 < argc) {
    nrounds = strtoull(argv[1], NULL, 0);
  }

  rjn_allocator *hdr = (rjn_allocator *)rjnbuf;
  rjn_init(hdr, sizeof(rjnbuf), 1 << 6, 1);

  pthread_t threads[4];
  for (int i = 0; i < NTHREADS; i++) {
    pthread_create(&threads[i], NULL, malloc_test_thread, hdr);
  }

  for (int i = 0; i < NTHREADS; i++) {
    pthread_join(threads[i], NULL);
  }

  assert(rjn_allocated(hdr) == 0);

  rjn_deinit(hdr);

  return 0;
}