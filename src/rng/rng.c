#include "hdcd/rng.h"

void hdcd_rng_seed(hdcd_rng_t *rng, uint64_t seed) {
    rng->state = seed ? seed : 0x9E3779B97F4A7C15ULL;
}

double hdcd_rng_uniform(hdcd_rng_t *rng) {
    rng->state ^= rng->state >> 12;
    rng->state ^= rng->state << 25;
    rng->state ^= rng->state >> 27;
    uint64_t result = rng->state * 0x2545F4914F6CDD1DULL;
    double u = (double)(result >> 11) * (1.0 / 9007199254740992.0); /* / 2^53 */
    if (u <= 0.0) u = 1e-15;
    if (u >= 1.0) u = 1.0 - 1e-15;
    return u;
}

void hdcd_rng_shuffle_indices(hdcd_rng_t *rng, size_t *indices, size_t n) {
    if (n < 2) {
        return;
    }
    for (size_t i = n - 1; i > 0; i--) {
        size_t j = (size_t)(hdcd_rng_uniform(rng) * (double)(i + 1));
        if (j > i) {
            j = i; /* guard against the extremely rare u -> 1.0 rounding edge */
        }
        size_t tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
    }
}
