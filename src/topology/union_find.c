#include "union_find.h"

#include <stdlib.h>

int hdcd_uf_init(hdcd_union_find_t *uf, size_t n) {
    uf->parent = (size_t *)malloc(n * sizeof(size_t));
    uf->size = (size_t *)malloc(n * sizeof(size_t));
    uf->n = n;
    if (uf->parent == NULL || uf->size == NULL) {
        free(uf->parent);
        free(uf->size);
        uf->parent = NULL;
        uf->size = NULL;
        return 0;
    }
    for (size_t i = 0; i < n; i++) {
        uf->parent[i] = i;
        uf->size[i] = 1;
    }
    return 1;
}

void hdcd_uf_free(hdcd_union_find_t *uf) {
    free(uf->parent);
    free(uf->size);
    uf->parent = NULL;
    uf->size = NULL;
}

size_t hdcd_uf_find(hdcd_union_find_t *uf, size_t x) {
    /* Iterative path halving: avoids recursion depth concerns. */
    while (uf->parent[x] != x) {
        uf->parent[x] = uf->parent[uf->parent[x]];
        x = uf->parent[x];
    }
    return x;
}

int hdcd_uf_union(hdcd_union_find_t *uf, size_t a, size_t b) {
    size_t ra = hdcd_uf_find(uf, a);
    size_t rb = hdcd_uf_find(uf, b);
    if (ra == rb) {
        return 0;
    }
    /* Union by size, tie broken toward the smaller index for
     * deterministic (if otherwise unobservable) internal behavior. */
    if (uf->size[ra] < uf->size[rb] || (uf->size[ra] == uf->size[rb] && rb < ra)) {
        size_t tmp = ra;
        ra = rb;
        rb = tmp;
    }
    uf->parent[rb] = ra;
    uf->size[ra] += uf->size[rb];
    return 1;
}
