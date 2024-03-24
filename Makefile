CC = gcc
LD = gcc
CFLAGS = -std=c11 -g -Wall -Wextra -Werror -Wno-unused-parameter -mlzcnt
LDFLAGS = -g
LDLIBS = -lm

rjntest: rjntest.o rjnmalloc.o allocator.o

clean:
	rm -f rjntest rjntest.o rjnmalloc.o allocator.o
