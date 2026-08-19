#ifndef HDCD_RNG_H
#define HDCD_RNG_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Minimal deterministic seeded RNG (xorshift64*), used wherever the
 * library needs reproducible randomness (spec section 24: "use
 * deterministic seeded RNG"; section 36 rule 14: "every stochastic
 * routine must accept an explicit seed"). Pulled forward from its
 * originally-scheduled module (spec section 22 lists rng/rng.c
 * alongside the annealing/sampling milestones) because Milestone 7's
 * held-out train/split scoring (spec section 16) needs a reproducible
 * shuffle now.
 */
typedef struct hdcd_rng {
    uint64_t state;
} hdcd_rng_t;

void hdcd_rng_seed(hdcd_rng_t *rng, uint64_t seed);

/* Uniform double in the open interval (0,1). */
double hdcd_rng_uniform(hdcd_rng_t *rng);

/* In-place Fisher-Yates shuffle of indices[0..n-1]. */
void hdcd_rng_shuffle_indices(hdcd_rng_t *rng, size_t *indices, size_t n);

#ifdef __cplusplus
}
#endif

#endif /* HDCD_RNG_H */
