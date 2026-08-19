# hdcd build: core numerics, Gaussian-mixture marginal smoother (Milestone 1),
# the copula transform x -> u (Milestone 2), pairwise distance correlation
# (Milestone 3), MST/persistent-topology ordering (Milestone 4), the
# centered Bernstein tensor basis (Milestone 5), Sinkhorn normalization
# (Milestone 6), fixed-DAG fitting (Milestone 7), simulated-annealing DAG
# search (Milestone 8), and arbitrary-DAG / held-out-KL comparison
# (Milestone 9). EVT tails and language bindings are excluded until their
# respective milestones (spec section 31).

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
  src/marginal/bandwidth_cv.c \
  src/marginal/marginal_model.c \
  src/copula/transform.c \
  src/dcor/dcor_exact.c \
  src/dcor/dependence_matrix.c \
  src/topology/union_find.c \
  src/topology/mst.c \
  src/topology/persistent_affinity.c \
  src/topology/merge_tree.c \
  src/topology/ordering.c \
  src/topology/topology.c \
  src/basis/bernstein.c \
  src/basis/centered_bernstein.c \
  src/basis/difference_penalty.c \
  src/numerics/quadrature.c \
  src/sinkhorn/normalize.c \
  src/rng/rng.c \
  src/dag/graph.c \
  src/optimize/local_fit.c \
  src/core/model.c \
  src/dag/cache.c \
  src/dag/proposals.c \
  src/optimize/annealing.c
LIB_OBJS := $(patsubst %.c,$(BUILD_DIR)/%.o,$(LIB_SRCS))
STATIC_LIB := $(BUILD_DIR)/libhdcd.a

TEST_SRCS := $(wildcard tests/test_*.c)
TEST_BINS := $(patsubst tests/%.c,$(BUILD_DIR)/tests/%,$(TEST_SRCS))

EXAMPLE_SRCS := $(wildcard examples/*.c)
EXAMPLE_BINS := $(patsubst examples/%.c,$(BUILD_DIR)/examples/%,$(EXAMPLE_SRCS))

.PHONY: all test examples clean

all: $(STATIC_LIB) $(TEST_BINS) $(EXAMPLE_BINS)

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

examples: $(EXAMPLE_BINS)

test: $(TEST_BINS)
	@set -e; \
	for t in $(TEST_BINS); do \
		echo "== $$t =="; \
		./$$t; \
	done

clean:
	rm -rf $(BUILD_DIR)
