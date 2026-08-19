#include "hdcd/rng.h"
#include "test_utils.h"

#include <stdlib.h>

static void test_uniform_in_range_and_reproducible(void) {
    hdcd_rng_t rng1, rng2;
    hdcd_rng_seed(&rng1, 42);
    hdcd_rng_seed(&rng2, 42);

    for (int i = 0; i < 1000; i++) {
        double u1 = hdcd_rng_uniform(&rng1);
        double u2 = hdcd_rng_uniform(&rng2);
        HDCD_CHECK(u1 > 0.0 && u1 < 1.0);
        HDCD_CHECK(u1 == u2); /* identical seed -> identical stream */
    }
    HDCD_PASS("rng uniform stays in (0,1) and is reproducible under a fixed seed");
}

static void test_different_seeds_differ(void) {
    hdcd_rng_t rng1, rng2;
    hdcd_rng_seed(&rng1, 1);
    hdcd_rng_seed(&rng2, 2);

    int any_diff = 0;
    for (int i = 0; i < 20; i++) {
        if (hdcd_rng_uniform(&rng1) != hdcd_rng_uniform(&rng2)) {
            any_diff = 1;
        }
    }
    HDCD_CHECK(any_diff);
    HDCD_PASS("different seeds produce different streams");
}

static void test_shuffle_is_permutation_and_reproducible(void) {
    size_t n = 30;
    size_t *a = (size_t *)malloc(n * sizeof(size_t));
    size_t *b = (size_t *)malloc(n * sizeof(size_t));
    for (size_t i = 0; i < n; i++) { a[i] = i; b[i] = i; }

    hdcd_rng_t rng1, rng2;
    hdcd_rng_seed(&rng1, 123);
    hdcd_rng_seed(&rng2, 123);
    hdcd_rng_shuffle_indices(&rng1, a, n);
    hdcd_rng_shuffle_indices(&rng2, b, n);

    for (size_t i = 0; i < n; i++) {
        HDCD_CHECK(a[i] == b[i]); /* reproducible */
    }

    /* Still a permutation of 0..n-1: every value appears exactly once. */
    int *seen = (int *)calloc(n, sizeof(int));
    for (size_t i = 0; i < n; i++) {
        HDCD_CHECK(a[i] < n);
        HDCD_CHECK(!seen[a[i]]);
        seen[a[i]] = 1;
    }

    int any_moved = 0;
    for (size_t i = 0; i < n; i++) {
        if (a[i] != i) any_moved = 1;
    }
    HDCD_CHECK(any_moved);

    free(a);
    free(b);
    free(seen);
    HDCD_PASS("shuffle produces a reproducible permutation");
}

int main(void) {
    test_uniform_in_range_and_reproducible();
    test_different_seeds_differ();
    test_shuffle_is_permutation_and_reproducible();
    printf("All rng tests passed.\n");
    return 0;
}
