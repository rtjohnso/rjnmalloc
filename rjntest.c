#include "rjnmalloc.h"
#define _XOPEN_SOURCE
#include <assert.h>
#include <immintrin.h>
#include <inttypes.h>
#include <malloc.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// The space used by the rjn allocator
char rjnbuf[1 << 30];

// A wrapper of malloc for benchmarking comparison.  We limit the amount of
// space it will allocate at one time to make it more comparable to the rjn
// allocator.
typedef struct bounded_default_allocator {
  size_t max_size;
  size_t allocated;
} bounded_default_allocator;

void *bounded_default_alloc(void *state, size_t alignment, size_t size) {
  bounded_default_allocator *a = (bounded_default_allocator *)state;
  if (a->allocated + size > a->max_size) {
    return NULL;
  }
  void *p = aligned_alloc(alignment, size);
  if (p) {
    __sync_fetch_and_add(&a->allocated, malloc_usable_size(p));
  }
  return p;
}

void bounded_default_free(void *state, void *ptr) {
  bounded_default_allocator *a = (bounded_default_allocator *)state;
  assert(a->allocated >= malloc_usable_size(ptr));
  __sync_fetch_and_sub(&a->allocated, malloc_usable_size(ptr));
  free(ptr);
}

void *bounded_default_realloc(void *state, void *ptr, size_t alignment,
                              size_t size) {
  bounded_default_allocator *a = (bounded_default_allocator *)state;
  size_t old_size = malloc_usable_size(ptr);
  assert(a->allocated >= old_size);
  if (a->allocated + size - old_size > a->max_size) {
    return NULL;
  }
  void *p = realloc(ptr, size);
  __sync_fetch_and_add(&a->allocated, malloc_usable_size(p) - old_size);
  return p;
}

allocator_ops bounded_default_allocator_ops = {
    bounded_default_alloc, bounded_default_free, bounded_default_realloc};

bounded_default_allocator bd_allocator = {sizeof(rjnbuf), 0};

//
// The test code itself
//

typedef struct allocation {
  uint8_t *p;
  uint64_t alignment;
  size_t size;
  int c;
} allocation;

typedef enum alignment_mode { NONE, ARBITRARY } alignment_mode;

typedef struct test_params {
  void *state;
  allocator_ops *ops;
  uint64_t nrounds;
  alignment_mode al;
  int check_contents;
  int64_t alloc_weight;
  int64_t realloc_weight;
  int64_t free_weight;
  uint64_t min_alloc_size;
  uint64_t max_alloc_size;
  uint64_t max_allocations;
  uint64_t num_allocations;
  allocation *allocations;
  uint64_t successful_allocations;
  uint64_t failed_allocations;
  uint64_t successful_reallocations;
  uint64_t failed_reallocations;
} test_params;

#define MIN(a, b) ((a) < (b) ? (a) : (b))

void check_contents(test_params *params, int i) {
  if (params->check_contents && params->allocations[i].p) {
    for (unsigned int j = 0; j < params->allocations[i].size; j++) {
      assert(params->allocations[i].p[j] == params->allocations[i].c);
    }
  }
}

void malloc_test(test_params *params) {
  void *hdr = params->state;
  allocator_ops *ops = params->ops;
  allocation *allocations = params->allocations;
  int total_weight =
      params->alloc_weight + params->realloc_weight + params->free_weight;
  assert(0 < total_weight);

  double lsize;
  if (params->max_alloc_size == params->min_alloc_size) {
    lsize = 0;
  } else {
    lsize = log(params->max_alloc_size - params->min_alloc_size + 1);
  }

  for (uint64_t j = 0; j < params->nrounds; j++) {
    // Choose our operation and some common parameters
    int op;
    if (params->num_allocations == 0) {
      if (params->alloc_weight + params->realloc_weight == 0) {
        op = -1; // Always alloc
      } else {
        op = rand() % (params->alloc_weight + params->realloc_weight);
      }
    } else if (params->num_allocations == params->max_allocations) {
      if (params->realloc_weight + params->free_weight == 0) {
        op = total_weight; // Always free
      } else {
        op = (rand() % (params->realloc_weight + params->free_weight)) +
             params->alloc_weight;
      }
    } else {
      op = rand() % total_weight;
    }
    uint64_t newsize =
        params->min_alloc_size + exp(lsize * drand48() * drand48() - 1);
    uint64_t newalignment =
        params->al == ARBITRARY ? exp(lsize * drand48() - 1) : sizeof(void *);

    // Pick an allocation to operate on and do the operation
    uint64_t i;
    if (op < params->alloc_weight) {
      // Alloc
      i = params->num_allocations;
      assert(i < params->max_allocations);
      allocations[i].p = ops->alloc(hdr, newalignment, newsize);
      if (allocations[i].p) {
        allocations[i].size = newsize;
        allocations[i].alignment = newalignment;
        params->successful_allocations++;
        params->num_allocations++;
      } else {
        params->failed_allocations++;
      }

    } else if (op < params->alloc_weight + params->realloc_weight) {
      // Realloc
      i = (rand() % (params->num_allocations + 1)) % params->max_allocations;
      assert(i < params->max_allocations);
      check_contents(params, i);
      uint8_t *newp =
          ops->realloc(hdr, allocations[i].p, newalignment, newsize);
      if (newp || newsize == 0) {
        allocations[i].p = newp;
        allocations[i].size = newsize;
        allocations[i].alignment = newalignment;
        params->successful_reallocations++;
      } else {
        params->failed_reallocations++;
      }

      if (i == params->num_allocations && allocations[i].p) {
        params->num_allocations++;
      } else if (i < params->num_allocations && allocations[i].p == NULL) {
        allocation tmp = allocations[i];
        allocations[i] = allocations[params->num_allocations - 1];
        allocations[params->num_allocations - 1] = tmp;
        params->num_allocations--;
        i = params->num_allocations;
      }

    } else {
      // Free
      i = rand() % params->num_allocations;
      assert(i < params->max_allocations);
      check_contents(params, i);
      ops->free(hdr, allocations[i].p);
      allocations[i].p = NULL;
      allocation tmp = allocations[i];
      allocations[i] = allocations[params->num_allocations - 1];
      allocations[params->num_allocations - 1] = tmp;
      params->num_allocations--;
      i = params->num_allocations;
    }

    // Fill the allocation with a known value
    if (i < params->num_allocations) {
      allocations[i].c = rand() % 256;
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
  for (uint64_t i = 0; i < params->num_allocations; i++) {
    if (params->check_contents) {
      for (unsigned int j = 0; j < allocations[i].size; j++) {
        assert(allocations[i].p[j] == allocations[i].c);
      }
    }
    ops->free(hdr, allocations[i].p);
    allocations[i].p = NULL;
  }
}

// User interface and main driver

__attribute__((noreturn)) void usage(char *argv0) {
  int argv0len = strlen(argv0);
  printf(
      "Usage: %1$s [-a allocator] [-c] [-n nallocs] [-r rounds] [-t nthreads]\n"
      "       %2$*3$s [-l alignment-mode] [-m min-allocation-size]\n"
      "       %2$*3$s [-M max-allocation-size] [-A alloc-weight]\n"
      "       %2$*3$s [-R realloc-weight] [-F free-weight] [-s seed]\n",
      argv0, "", argv0len);
  printf("  -a: allocator (\"malloc\", \"rjn\")\n");
  printf("  -c: check contents of allocations\n");
  printf("  -n: number of allocations\n");
  printf("  -r: number of rounds\n");
  printf("  -t: number of threads\n");
  printf("  -l: alignment mode (\"none\", \"arbitrary\")\n");
  printf("  -m: minimum allocation size\n");
  printf("  -M: maximum allocation size\n");
  printf("  -A: weight of allocations\n");
  printf("  -R: weight of reallocations\n");
  printf("  -F: weight of frees\n");
  printf("  -s: seed\n");
  exit(EXIT_FAILURE);
}

int main(int argc, char **argv) {
  int nallocs = 10000;
  int nrounds = 1000000;
  int nthreads = 8;
  int check_contents = 0;
  uint64_t seed = 0;
  char *allocator = "rjn";
  alignment_mode alignment = NONE;
  uint64_t min_alloc_size = 0;
  uint64_t max_alloc_size = 0;
  uint64_t alloc_weight = 2;
  uint64_t realloc_weight = 1;
  uint64_t free_weight = 1;

  int opt;
  char *endptr;
  while ((opt = getopt(argc, argv, ":a:cn:r:t:l:m:M:A:R:F:s:")) != -1) {
    switch (opt) {
    case 'a':
      allocator = optarg;
      break;
    case 'c':
      check_contents = 1;
      break;
    case 'n':
      nallocs = strtoull(optarg, &endptr, 0);
      if (*endptr != '\0') {
        printf("Invalid number of allocations: %s\n", optarg);
        usage(argv[0]);
      }
      break;
    case 'r':
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
    case 'm':
      min_alloc_size = strtoull(optarg, &endptr, 0);
      if (*endptr != '\0') {
        printf("Invalid min allocation size: %s\n", optarg);
        usage(argv[0]);
      }
      break;
    case 'M':
      max_alloc_size = strtoull(optarg, &endptr, 0);
      if (*endptr != '\0') {
        printf("Invalid max allocation size: %s\n", optarg);
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
    case 'A':
      alloc_weight = strtoull(optarg, &endptr, 0);
      if (*endptr != '\0') {
        printf("Invalid allocation weight: %s\n", optarg);
        usage(argv[0]);
      }
      break;
    case 'R':
      realloc_weight = strtoull(optarg, &endptr, 0);
      if (*endptr != '\0') {
        printf("Invalid reallocation weight: %s\n", optarg);
        usage(argv[0]);
      }
      break;
    case 'F':
      free_weight = strtoull(optarg, &endptr, 0);
      if (*endptr != '\0') {
        printf("Invalid free weight: %s\n", optarg);
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

  if (max_alloc_size == 0) {
    max_alloc_size = sizeof(rjnbuf);
  }

  if (max_alloc_size < min_alloc_size) {
    printf(
        "Max allocation size must be greater than or equal to min allocation "
        "size\n");
    usage(argv[0]);
  }

  if (alloc_weight + realloc_weight + free_weight == 0) {
    printf("At least one of alloc_weight, realloc_weight, or free_weight must "
           "be non-zero\n");
    usage(argv[0]);
  }

  test_params *params = (test_params *)malloc(sizeof(test_params) * nthreads);

  for (int i = 0; i < nthreads; i++) {
    params[i].state = state;
    params[i].ops = ops;
    params[i].nrounds = nrounds;
    params[i].al = alignment;
    params[i].check_contents = check_contents;
    params[i].min_alloc_size = min_alloc_size;
    params[i].max_alloc_size = max_alloc_size;
    params[i].alloc_weight = alloc_weight;
    params[i].realloc_weight = realloc_weight;
    params[i].free_weight = free_weight;
    params[i].max_allocations = nallocs;
    params[i].num_allocations = 0;
    params[i].allocations = (allocation *)calloc(nallocs, sizeof(allocation));

    params[i].successful_allocations = 0;
    params[i].failed_allocations = 0;
    params[i].successful_reallocations = 0;
    params[i].failed_reallocations = 0;
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