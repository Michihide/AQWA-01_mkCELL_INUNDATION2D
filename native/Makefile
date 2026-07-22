CC ?= cc
CFLAGS ?= -O3 -fPIC -Wall -Wextra -std=c11
LDFLAGS ?= -shared

# OpenMP for quadtree refinement (Linux/gcc). macOS clang needs libomp; disabled on Darwin.
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
USE_OPENMP = 0
else
USE_OPENMP ?= 1
endif
ifeq ($(USE_OPENMP),1)
    CFLAGS += -fopenmp
    LDFLAGS += -fopenmp
endif

ifeq ($(shell uname),Darwin)
    LDFLAGS += -undefined dynamic_lookup
endif

SRCS = raster_zonal.c quadtree.c
OBJS = $(SRCS:.c=.o)
ifeq ($(UNAME_S),Darwin)
LIB = libmkcell_native.dylib
else
LIB = libmkcell_native.so
endif

.PHONY: all clean

all: $(LIB)

$(LIB): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $(OBJS)

%.o: %.c mkcell_native.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(LIB)
