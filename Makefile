CC = gcc
LD = gcc
CFLAGS = -g -Wall -Wextra -Werror -std=c99 -pedantic
LDFLAGS = -lm

rjntest: rjntest.o rjnmalloc.o

clean:
	rm -f rjntest rjntest.o rjnmalloc.o
