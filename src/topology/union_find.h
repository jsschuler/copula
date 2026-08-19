#ifndef HDCD_INTERNAL_UNION_FIND_H
#define HDCD_INTERNAL_UNION_FIND_H

#include <stddef.h>

/* Plain union-find (disjoint-set), private to the topology module. */
typedef struct {
    size_t *parent;
    size_t *size;
    size_t n;
} hdcd_union_find_t;

/* Returns 1 on success, 0 on allocation failure. */
int hdcd_uf_init(hdcd_union_find_t *uf, size_t n);
void hdcd_uf_free(hdcd_union_find_t *uf);

size_t hdcd_uf_find(hdcd_union_find_t *uf, size_t x);

/* Union by size, with path compression on find. Returns 1 if a union
 * was actually performed (a and b were in different sets), 0 if they
 * were already in the same set. */
int hdcd_uf_union(hdcd_union_find_t *uf, size_t a, size_t b);

#endif /* HDCD_INTERNAL_UNION_FIND_H */
