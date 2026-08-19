#include "cache.h"

#include <stdlib.h>

typedef struct {
    size_t child;
    size_t n_parents;
    size_t *sorted_parents; /* owned, ascending */
    hdcd_local_fit_t *fit;   /* owned */
    hdcd_status_t status;     /* status hdcd_local_fit_node returned when this was fitted */
} cache_entry_t;

struct hdcd_local_fit_cache {
    size_t d;
    cache_entry_t **entries;   /* entries[child] = dynamic array */
    size_t *count;
    size_t *capacity;
};

static int compare_size_t(const void *a, const void *b) {
    size_t va = *(const size_t *)a;
    size_t vb = *(const size_t *)b;
    if (va < vb) return -1;
    if (va > vb) return 1;
    return 0;
}

hdcd_status_t hdcd_internal_cache_create(size_t d, hdcd_local_fit_cache_t **out) {
    if (out == NULL || d == 0) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }
    *out = NULL;

    hdcd_local_fit_cache_t *cache = (hdcd_local_fit_cache_t *)malloc(sizeof(hdcd_local_fit_cache_t));
    if (cache == NULL) {
        return HDCD_ERROR_ALLOCATION;
    }
    cache->d = d;
    cache->entries = (cache_entry_t **)calloc(d, sizeof(cache_entry_t *));
    cache->count = (size_t *)calloc(d, sizeof(size_t));
    cache->capacity = (size_t *)calloc(d, sizeof(size_t));
    if (cache->entries == NULL || cache->count == NULL || cache->capacity == NULL) {
        free(cache->entries);
        free(cache->count);
        free(cache->capacity);
        free(cache);
        return HDCD_ERROR_ALLOCATION;
    }

    *out = cache;
    return HDCD_OK;
}

void hdcd_internal_cache_free(hdcd_local_fit_cache_t *cache) {
    if (cache == NULL) {
        return;
    }
    for (size_t c = 0; c < cache->d; c++) {
        for (size_t i = 0; i < cache->count[c]; i++) {
            free(cache->entries[c][i].sorted_parents);
            hdcd_local_fit_free(cache->entries[c][i].fit);
        }
        free(cache->entries[c]);
    }
    free(cache->entries);
    free(cache->count);
    free(cache->capacity);
    free(cache);
}

static cache_entry_t *find_entry(
    hdcd_local_fit_cache_t *cache, size_t child, const size_t *sorted_parents, size_t n_parents
) {
    for (size_t i = 0; i < cache->count[child]; i++) {
        cache_entry_t *e = &cache->entries[child][i];
        if (e->n_parents != n_parents) {
            continue;
        }
        int match = 1;
        for (size_t k = 0; k < n_parents; k++) {
            if (e->sorted_parents[k] != sorted_parents[k]) {
                match = 0;
                break;
            }
        }
        if (match) {
            return e;
        }
    }
    return NULL;
}

hdcd_status_t hdcd_internal_cache_get_or_fit(
    hdcd_local_fit_cache_t *cache,
    const double *u, const uint8_t *mask, size_t n, size_t d,
    size_t child, const size_t *parents, size_t n_parents,
    const hdcd_local_fit_options_t *options,
    const hdcd_local_fit_t **out,
    int *was_hit
) {
    if (cache == NULL || out == NULL || child >= cache->d) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }
    if (n_parents > 0 && parents == NULL) {
        return HDCD_ERROR_INVALID_ARGUMENT;
    }
    *out = NULL;
    if (was_hit != NULL) {
        *was_hit = 0;
    }

    size_t *sorted_parents = NULL;
    if (n_parents > 0) {
        sorted_parents = (size_t *)malloc(n_parents * sizeof(size_t));
        if (sorted_parents == NULL) {
            return HDCD_ERROR_ALLOCATION;
        }
        for (size_t i = 0; i < n_parents; i++) {
            sorted_parents[i] = parents[i];
        }
        qsort(sorted_parents, n_parents, sizeof(size_t), compare_size_t);
    }

    cache_entry_t *existing = find_entry(cache, child, sorted_parents, n_parents);
    if (existing != NULL) {
        free(sorted_parents);
        if (was_hit != NULL) {
            *was_hit = 1;
        }
        *out = existing->fit;
        return existing->status;
    }

    hdcd_local_fit_t *fit = NULL;
    hdcd_status_t status = hdcd_local_fit_node(u, mask, n, d, child, parents, n_parents, options, &fit);
    if (fit == NULL) {
        free(sorted_parents);
        return status; /* hard failure: not cached */
    }

    if (cache->count[child] == cache->capacity[child]) {
        size_t new_cap = (cache->capacity[child] == 0) ? 4 : cache->capacity[child] * 2;
        cache_entry_t *grown = (cache_entry_t *)realloc(cache->entries[child], new_cap * sizeof(cache_entry_t));
        if (grown == NULL) {
            free(sorted_parents);
            hdcd_local_fit_free(fit);
            return HDCD_ERROR_ALLOCATION;
        }
        cache->entries[child] = grown;
        cache->capacity[child] = new_cap;
    }

    cache_entry_t *slot = &cache->entries[child][cache->count[child]++];
    slot->child = child;
    slot->n_parents = n_parents;
    slot->sorted_parents = sorted_parents;
    slot->fit = fit;
    slot->status = status;

    *out = fit;
    return status;
}
