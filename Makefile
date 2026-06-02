CC      = gcc
CFLAGS  = -Wall -Wextra -std=c11 -O2 -D_GNU_SOURCE
LDFLAGS = -lm

SRCDIR  = src
INCDIR  = include
TESTDIR = tests

SRCS    = $(SRCDIR)/hodge.c $(SRCDIR)/belief.c $(SRCDIR)/interpretability.c
OBJS    = $(SRCS:.c=.o)
TESTSRC = $(TESTDIR)/test_hodge_belief.c

.PHONY: all test clean

all: libhodgebelief.a

%.o: %.c
	$(CC) $(CFLAGS) -I$(INCDIR) -c $< -o $@

libhodgebelief.a: $(OBJS)
	ar rcs $@ $^

test_bin: $(TESTSRC) libhodgebelief.a
	$(CC) $(CFLAGS) -I$(INCDIR) $(TESTSRC) -L. -lhodgebelief $(LDFLAGS) -o $@

test: test_bin
	./test_bin

clean:
	rm -f $(SRCDIR)/*.o libhodgebelief.a test_bin
