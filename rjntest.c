#include "rjnmalloc.h"
#define _XOPEN_SOURCE
#include <assert.h>
#include <inttypes.h>
#include <malloc.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct bounded_default_allocator {
  size_t max_size;
  size_t allocated;
} bounded_default_allocator;

void *bounded_default_alloc(void *state, size_t alignment, size_t size) {
  bounded_default_allocator *a = (bounded_default_allocator *)state;
  if (a->allocated + size > a->max_size) {
    return NULL;
  }
  a->allocated += size;
  return aligned_alloc(alignment, size);
}

void bounded_default_free(void *state, void *ptr) {
  bounded_default_allocator *a = (bounded_default_allocator *)state;
  a->allocated -= malloc_usable_size(ptr);
  free(ptr);
}

void *bounded_default_realloc(void *state, void *ptr, size_t alignment,
                              size_t size) {
  bounded_default_allocator *a = (bounded_default_allocator *)state;
  if (a->allocated + size - malloc_usable_size(ptr) > a->max_size) {
    return NULL;
  }
  a->allocated += size;
  return realloc(ptr, size);
}

allocator_ops bounded_default_allocator_ops = {
    bounded_default_alloc, bounded_default_free, bounded_default_realloc};

char rjnbuf[1 << 30];

bounded_default_allocator bd_allocator = {sizeof(rjnbuf), 0};

typedef struct allocation {
  uint8_t *p;
  uint64_t alignment;
  size_t size;
  int c;
} allocation;

#define NALLOCATIONS (1000)

typedef enum alignment_mode { NONE, ARBITRARY } alignment_mode;

typedef struct test_params {
  void *state;
  allocator_ops *ops;
  uint64_t nrounds;
  alignment_mode al;
  uint64_t successful_allocations;
  uint64_t failed_allocations;
  uint64_t successful_reallocations;
  uint64_t failed_reallocations;
  int check_contents;
  allocation allocations[NALLOCATIONS];
} test_params;

#define MIN(a, b) ((a) < (b) ? (a) : (b))

void malloc_test(test_params *params) {
  void *hdr = params->state;
  allocator_ops *ops = params->ops;
  allocation *allocations = params->allocations;
  memset(params->allocations, 0, sizeof(params->allocations));

  double lsize = log(sizeof(rjnbuf));

  for (uint64_t j = 0; j < params->nrounds; j++) {
    int i = rand() % NALLOCATIONS;
    if (allocations[i].p && params->check_contents) {
      for (unsigned int k = 0; k < allocations[i].size; k++) {
        assert(allocations[i].p[k] == allocations[i].c);
      }
    }

    uint64_t newsize = exp(lsize * drand48());
    uint64_t newalignment =
        params->al == ARBITRARY ? exp(lsize * drand48()) : sizeof(void *);

    if (rand() % 2) {
      uint8_t *newp =
          ops->realloc(hdr, allocations[i].p, newalignment, newsize);
      if (newp || allocations[i].size == 0) {
        allocations[i].p = newp;
        allocations[i].size = newsize;
        allocations[i].alignment = newalignment;
        params->successful_reallocations++;
      } else {
        params->failed_reallocations++;
      }
    } else if (allocations[i].p) {
      ops->free(hdr, allocations[i].p);
      allocations[i].p = NULL;
    } else {
      allocations[i].p = ops->alloc(hdr, newalignment, newsize);
      if (allocations[i].p) {
        allocations[i].size = newsize;
        allocations[i].alignment = newalignment;
        params->successful_allocations++;
      } else {
        params->failed_allocations++;
      }
    }
    allocations[i].c = rand() % 256;
    if (allocations[i].p) {
      if (allocations[i].alignment) {
        assert(((uintptr_t)allocations[i].p) % allocations[i].alignment == 0);
      }
      if (params->check_contents) {
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
  void *hdr = params->state;
  allocator_ops *ops = params->ops;
  allocation *allocations = params->allocations;
  for (int i = 0; i < NALLOCATIONS; i++) {
    if (allocations[i].p) {
      if (params->check_contents) {
        for (unsigned int j = 0; j < allocations[i].size; j++) {
          assert(allocations[i].p[j] == allocations[i].c);
        }
      }
      ops->free(hdr, allocations[i].p);
      allocations[i].p = NULL;
    }
  }
}

__attribute__((noreturn)) void usage(char *argv0) {
  printf("Usage: %s [-a allocator] [-c] [-n nrounds] [-t nthreads] [-l "
         "alignment-mode] [-s seed]\n",
         argv0);
  printf("  -a: allocator (\"malloc\", \"rjn\")\n");
  printf("  -c: check contents of allocations\n");
  printf("  -n: number of rounds\n");
  printf("  -t: number of threads\n");
  printf("  -l: alignment mode (\"none\", \"arbitrary\")\n");
  printf("  -s: seed\n");
  exit(EXIT_FAILURE);
}

int main(int argc, char **argv) {
  int nrounds = 1000000;
  int nthreads = 4;
  int check_contents = 0;
  uint64_t seed = 0;
  char *allocator = "rjn";
  alignment_mode alignment = NONE;

  int opt;
  char *endptr;
  while ((opt = getopt(argc, argv, ":a:cn:t:l:s:")) != -1) {
    switch (opt) {
    case 'a':
      allocator = optarg;
      break;
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
    case 'l':
      if (strcmp(optarg, "none") == 0) {
        alignment = NONE;
      } else if (strcmp(optarg, "arbitrary") == 0) {
        alignment = ARBITRARY;
      } else {
        printf("Invalid alignment mode: %s\n", optarg);
        usage(argv[0]);
      }
      break;
    case 's':
      seed = strtoull(optarg, &endptr, 0);
      if (*endptr != '\0') {
        printf("Invalid seed: %s\n", optarg);
        usage(argv[0]);
      }
      srand(seed);
      srand48(rand());
      break;
    case ':':
      printf("Option -%c requires an operand\n", optopt);
      usage(argv[0]);
    default:
      printf("Unknown option: -%c\n", optopt);
      usage(argv[0]);
    }
  }

  allocator_ops *ops = NULL;
  void *state = NULL;
  if (strcmp(allocator, "malloc") == 0) {
    ops = &bounded_default_allocator_ops;
    state = &bd_allocator;
  } else if (strcmp(allocator, "rjn") == 0) {
    ops = &rjn_allocator_ops;
    rjn *hdr = (rjn *)rjnbuf;
    int r = rjn_init(hdr, sizeof(rjnbuf), 1 << 6);
    assert(r == 0);
    state = hdr;
  } else {
    printf("Unknown allocator: %s\n", allocator);
    usage(argv[0]);
  }

  test_params *params = (test_params *)malloc(sizeof(test_params) * nthreads);

  for (int i = 0; i < nthreads; i++) {
    params[i].state = state;
    params[i].ops = ops;
    params[i].nrounds = nrounds;
    params[i].al = alignment;
    params[i].successful_allocations = 0;
    params[i].failed_allocations = 0;
    params[i].successful_reallocations = 0;
    params[i].failed_reallocations = 0;
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

  if (ops == &rjn_allocator_ops) {
    printf("Allocation stats after test:\n");
    rjn_print_allocation_stats((rjn *)state);
  }

  uint64_t total_successful_allocations = 0;
  uint64_t total_failed_allocations = 0;
  uint64_t total_successful_reallocations = 0;
  uint64_t total_failed_reallocations = 0;
  for (int i = 0; i < nthreads; i++) {
    total_successful_allocations += params[i].successful_allocations;
    total_failed_allocations += params[i].failed_allocations;
    total_successful_reallocations += params[i].successful_reallocations;
    total_failed_reallocations += params[i].failed_reallocations;
  }
  printf("Total successful allocations: %" PRIu64 "\n",
         total_successful_allocations);
  printf("Total failed allocations: %" PRIu64 "\n", total_failed_allocations);
  printf("Total successful reallocations: %" PRIu64 "\n",
         total_successful_reallocations);
  printf("Total failed reallocations: %" PRIu64 "\n",
         total_failed_reallocations);

  printf("Cleaning up\n");

  for (int i = 0; i < nthreads; i++) {
    cleanup_test(&params[i]);
  }

  if (ops == &rjn_allocator_ops) {
    printf("Allocation stats after cleanup:\n");
    rjn_print_allocation_stats(state);
    rjn_deinit(state);
  }

  return 0;
}