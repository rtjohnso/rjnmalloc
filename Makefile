CC = gcc
LD = gcc
CFLAGS = -O3 -std=c11 -g -Wall -Wextra -Werror -Wno-unused-parameter -mlzcnt
LDFLAGS = -O3 -g
LDLIBS = -lm

rjntest: rjntest.o rjnmalloc.o allocator.o

clean:
	rm -f rjntest rjntest.o rjnmalloc.o allocator.o
