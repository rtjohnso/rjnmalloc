CC = gcc
LD = gcc
CFLAGS = -g -Wall -Wextra -Werror -Wno-unused-parameter -std=c99 -pedantic -mlzcnt
LDLIBS = -lm

rjntest: rjntest.o rjnmalloc.o

clean:
	rm -f rjntest rjntest.o rjnmalloc.o
