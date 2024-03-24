CC = clang
LD = clang
CFLAGS = -O3 -g -Wall -Wextra -Werror -Wno-unused-parameter -std=c99 -mlzcnt
LDFLAGS = -O3 -g
LDLIBS = -lm

rjntest: rjntest.o rjnmalloc.o

clean:
	rm -f rjntest rjntest.o rjnmalloc.o
