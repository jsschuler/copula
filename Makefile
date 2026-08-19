# hdcd Milestone 1 build: core numerics + Gaussian-mixture marginal smoother.
# DAGs, distance correlation, topology, Bernstein basis, Sinkhorn
# normalization, EVT tails, and language bindings are excluded until
# their respective milestones (spec section 31).

CC ?= cc
CFLAGS ?= -std=c99 -Wall -Wextra -Wpedantic -O2 -g
INCLUDES := -Iinclude
LDLIBS := -lm

BUILD_DIR := build
LIB_SRCS := \
  src/core/errors.c \
  src/numerics/logsumexp.c \
  src/numerics/robust_scale.c \
  src/numerics/optimizer_1d.c \
  src/marginal/gaussian_cdf_mix.c \
  src/marginal/gaussian_density_mix.c \
  src/marginal/bandwidth_cv.c
LIB_OBJS := $(patsubst %.c,$(BUILD_DIR)/%.o,$(LIB_SRCS))
STATIC_LIB := $(BUILD_DIR)/libhdcd.a

TEST_SRCS := $(wildcard tests/test_*.c)
TEST_BINS := $(patsubst tests/%.c,$(BUILD_DIR)/tests/%,$(TEST_SRCS))

EXAMPLE_BIN := $(BUILD_DIR)/examples/example_fit_marginal

.PHONY: all test examples clean

all: $(STATIC_LIB) $(TEST_BINS) $(EXAMPLE_BIN)

$(BUILD_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(STATIC_LIB): $(LIB_OBJS)
	@mkdir -p $(dir $@)
	ar rcs $@ $^

$(BUILD_DIR)/tests/%: tests/%.c $(STATIC_LIB)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(INCLUDES) -Itests $< $(STATIC_LIB) $(LDLIBS) -o $@

$(BUILD_DIR)/examples/%: examples/%.c $(STATIC_LIB)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(INCLUDES) $< $(STATIC_LIB) $(LDLIBS) -o $@

examples: $(EXAMPLE_BIN)

test: $(TEST_BINS)
	@set -e; \
	for t in $(TEST_BINS); do \
		echo "== $$t =="; \
		./$$t; \
	done

clean:
	rm -rf $(BUILD_DIR)
