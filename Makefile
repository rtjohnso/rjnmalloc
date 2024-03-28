CC = gcc
LD = gcc
CFLAGS = -std=c11 -g -O3 -Wall -Wextra -Werror -Wno-unused-parameter -mlzcnt
LDFLAGS = -g -O3
LDLIBS = -lm

rjntest: rjntest.o rjnmalloc.o allocator.o

clean:
	rm -f rjntest rjntest.o rjnmalloc.o allocator.o
