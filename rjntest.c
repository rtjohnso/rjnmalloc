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
  uint64_t alignment;
  size_t size;
  int c;
} allocation;

#define NALLOCATIONS (1000)

typedef struct test_params {
  rjn_allocator *hdr;
  uint64_t nrounds;
  uint64_t successful_allocations;
  uint64_t failed_allocations;
  int check_contents;
  allocation allocations[NALLOCATIONS];
} test_params;

#define MIN(a, b) ((a) < (b) ? (a) : (b))

void malloc_test(test_params *params) {
  rjn_allocator *hdr = params->hdr;
  allocation *allocations = params->allocations;
  memset(params->allocations, 0, sizeof(params->allocations));

  double lsize = log(rjn_size(hdr));
  uint8_t *rstart = (uint8_t *)rjn_start(hdr);
  uint8_t *rend = (uint8_t *)rjn_end(hdr);
  size_t au_size = rjn_allocation_unit_size(hdr);

  for (uint64_t j = 0; j < params->nrounds; j++) {
    int i = rand() % NALLOCATIONS;
    if (allocations[i].p) {
      if (params->check_contents) {
        for (unsigned int k = 0; k < allocations[i].size; k++) {
          assert(allocations[i].p[k] == allocations[i].c);
        }
      }
      rjn_free(hdr, allocations[i].p);
      allocations[i].p = NULL;
    } else {
      allocations[i].size = exp(lsize * drand48());
      allocations[i].alignment = exp(lsize * drand48());
      allocations[i].p =
          rjn_alloc(hdr, allocations[i].alignment, allocations[i].size);
      allocations[i].c = rand() % 256;
      if (allocations[i].p) {
        params->successful_allocations++;
        assert(rstart <= allocations[i].p);
        assert(allocations[i].p + allocations[i].size <= rend);
        if (allocations[i].alignment) {
          assert(((uintptr_t)allocations[i].p) % allocations[i].alignment == 0);
        }
        if (params->check_contents) {
          memset(allocations[i].p, allocations[i].c, allocations[i].size);
        } else {
          memset(allocations[i].p, allocations[i].c,
                 MIN(allocations[i].size, au_size));
        }
      } else {
        params->failed_allocations++;
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
      if (params->check_contents) {
        for (unsigned int j = 0; j < allocations[i].size; j++) {
          assert(allocations[i].p[j] == allocations[i].c);
        }
      }
      rjn_free(hdr, allocations[i].p);
      allocations[i].p = NULL;
    }
  }
}

char rjnbuf[1 << 30];

__attribute__((noreturn)) void usage(char *argv0) {
  printf("Usage: %s [-c] [-n nrounds] [-t nthreads]\n", argv0);
  exit(EXIT_FAILURE);
}

int main(int argc, char **argv) {
  int nrounds = 1000000;
  int nthreads = 4;
  int check_contents = 0;

  int opt;
  char *endptr;
  while ((opt = getopt(argc, argv, ":cn:t:")) != -1) {
    switch (opt) {
    case 'c':
      check_contents = 1;
      break;
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
    params[i].successful_allocations = 0;
    params[i].failed_allocations = 0;
    params[i].check_contents = check_contents;
  }

  printf("Running malloc_test with %d thread%s for %d round%s\n", nthreads,
         nthreads == 1 ? "" : "s", nrounds, nrounds == 1 ? "" : "s");

  if (1 < nthreads) {
    pthread_t *threads = (pthread_t *)malloc(sizeof(pthread_t) * nthreads);
    for (int i = 0; i < nthreads; i++) {
      pthread_create(&threads[i], NULL, malloc_test_thread, &params[i]);
    }

    for (int i = 0; i < nthreads; i++) {
      pthread_join(threads[i], NULL);
    }
  } else {
    malloc_test(&params[0]);
  }

  printf("Allocation stats after test:\n");

  rjn_print_allocation_stats(hdr);

  uint64_t total_successful_allocations = 0;
  uint64_t total_failed_allocations = 0;
  for (int i = 0; i < nthreads; i++) {
    total_successful_allocations += params[i].successful_allocations;
    total_failed_allocations += params[i].failed_allocations;
  }
  printf("Total successful allocations: %" PRIu64 "\n",
         total_successful_allocations);
  printf("Total failed allocations: %" PRIu64 "\n", total_failed_allocations);

  printf("Cleaning up\n");

  for (int i = 0; i < nthreads; i++) {
    cleanup_test(&params[i]);
  }

  printf("Allocation stats after cleanup:\n");

  rjn_print_allocation_stats(hdr);

  rjn_deinit(hdr);

  return 0;
}