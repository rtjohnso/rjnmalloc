#include "rjnmalloc.h"
#define _XOPEN_SOURCE
#include <assert.h>
#include <inttypes.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct allocation {
  uint8_t *p;
  size_t size;
  int c;
} allocation;

#define NALLOCATIONS (1000)

typedef struct test_params {
  rjn_allocator *hdr;
  uint64_t nrounds;
  allocation allocations[NALLOCATIONS];
} test_params;

void malloc_test(test_params *params) {
  rjn_allocator *hdr = params->hdr;
  allocation *allocations = params->allocations;
  memset(params->allocations, 0, sizeof(params->allocations));

  double lsize = log(rjn_size(hdr));
  uint8_t *rstart = (uint8_t *)rjn_start(hdr);
  uint8_t *rend = (uint8_t *)rjn_end(hdr);

  for (uint64_t i = 0; i < params->nrounds; i++) {
    int i = rand() % NALLOCATIONS;
    if (allocations[i].p) {
      for (unsigned int j = 0; j < allocations[i].size; j++) {
        assert(allocations[i].p[j] == allocations[i].c);
      }
      rjn_free(hdr, allocations[i].p);
      allocations[i].p = NULL;
    } else {
      allocations[i].size = exp(lsize * drand48());
      allocations[i].p = rjn_alloc(hdr, 0, allocations[i].size);
      allocations[i].c = rand() % 256;
      if (allocations[i].p) {
        assert(rstart <= allocations[i].p);
        assert(allocations[i].p + allocations[i].size <= rend);
        memset(allocations[i].p, allocations[i].c, allocations[i].size);
      }
    }
  }
}

void *malloc_test_thread(void *p) {
  malloc_test((test_params *)p);
  return NULL;
}

void cleanup_test(test_params *params) {
  rjn_allocator *hdr = params->hdr;
  allocation *allocations = params->allocations;
  for (int i = 0; i < NALLOCATIONS; i++) {
    if (allocations[i].p) {
      for (unsigned int j = 0; j < allocations[i].size; j++) {
        assert(allocations[i].p[j] == allocations[i].c);
      }
      rjn_free(hdr, allocations[i].p);
      allocations[i].p = NULL;
    }
  }
}

char rjnbuf[1 << 30];

__attribute__((noreturn)) void usage(char *argv0) {
  printf("Usage: %s [-n nrounds] [-t nthreads]\n", argv0);
  exit(EXIT_FAILURE);
}

int main(int argc, char **argv) {
  int nrounds = 1000000;
  int nthreads = 4;

  int opt;
  char *endptr;
  while ((opt = getopt(argc, argv, ":n:t:")) != -1) {
    switch (opt) {
    case 'n':
      nrounds = strtoull(optarg, &endptr, 0);
      if (*endptr != '\0') {
        printf("Invalid number of rounds: %s\n", optarg);
        usage(argv[0]);
      }
      break;
    case 't':
      nthreads = strtoull(optarg, &endptr, 0);
      if (*endptr != '\0') {
        printf("Invalid number of threads: %s\n", optarg);
        usage(argv[0]);
      }
      break;
    case ':':
      printf("Option -%c requires an operand\n", optopt);
      usage(argv[0]);
    default:
      printf("Unknown option: -%c\n", optopt);
      usage(argv[0]);
    }
  }

  test_params *params = (test_params *)malloc(sizeof(test_params) * nthreads);

  rjn_allocator *hdr = (rjn_allocator *)rjnbuf;
  int r = rjn_init(hdr, sizeof(rjnbuf), 1 << 6);
  assert(r == 0);

  for (int i = 0; i < nthreads; i++) {
    params[i].hdr = hdr;
    params[i].nrounds = nrounds;
  }

  printf("Running malloc_test with %d threads for %d rounds\n", nthreads,
         nrounds);

  pthread_t *threads = (pthread_t *)malloc(sizeof(pthread_t) * nthreads);
  for (int i = 0; i < nthreads; i++) {
    pthread_create(&threads[i], NULL, malloc_test_thread, &params[i]);
  }

  for (int i = 0; i < nthreads; i++) {
    pthread_join(threads[i], NULL);
  }

  printf("Allocation stats after test:\n");

  rjn_print_allocation_stats(hdr);

  printf("Cleaning up\n");

  for (int i = 0; i < nthreads; i++) {
    cleanup_test(&params[i]);
  }

  printf("Allocation stats after cleanup:\n");

  rjn_print_allocation_stats(hdr);

  rjn_deinit(hdr);

  return 0;
}