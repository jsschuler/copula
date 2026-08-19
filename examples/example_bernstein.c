#include "hdcd/hdcd.h"

#include <stdio.h>
#include <stdlib.h>

/*
 * Milestone 5 example: evaluate the centered Bernstein tensor kernel
 * g(u,z) = B~(u)^T Theta B~(z) (spec section 9) for a hand-picked edge
 * coefficient surface, and report its roughness penalty (spec section
 * 10) before and after smoothing Theta.
 */

int main(void) {
    const size_t m = 4, dim = m + 1;
    double *theta_rough = (double *)malloc(dim * dim * sizeof(double));
    double *theta_smooth = (double *)malloc(dim * dim * sizeof(double));

    /* A "rough" surface: alternating sign, high second differences. */
    for (size_t r = 0; r < dim; r++) {
        for (size_t s = 0; s < dim; s++) {
            double sign = ((r + s) % 2 == 0) ? 1.0 : -1.0;
            theta_rough[r * dim + s] = sign * 1.5;
        }
    }
    /* A "smooth" surface: affine in each index separately (plus a
     * bilinear r*s cross term) has zero second difference along both
     * axes, so R = 0 -- but unlike a purely additive f(r)+g(s) surface
     * (which the centered basis would annihilate entirely, since the
     * centered basis sums to zero along each axis), the r*s term is not
     * additively separable, so g(u,z) is still generally nonzero. */
    for (size_t r = 0; r < dim; r++) {
        for (size_t s = 0; s < dim; s++) {
            theta_smooth[r * dim + s] = 0.15 * (double)r * (double)s + 0.3 * (double)r - 0.2 * (double)s;
        }
    }

    double r_rough, r_smooth;
    hdcd_bernstein_roughness_penalty(theta_rough, m, &r_rough);
    hdcd_bernstein_roughness_penalty(theta_smooth, m, &r_smooth);

    printf("hdcd Milestone 5 example: Bernstein tensor kernel and roughness penalty\n");
    printf("degree m = %zu\n\n", m);
    printf("R(theta_rough)  = %.4f\n", r_rough);
    printf("R(theta_smooth) = %.4f  (affine-in-index theta has zero second difference)\n\n", r_smooth);

    printf("%8s %8s %14s %14s\n", "u", "z", "g(u,z) rough", "g(u,z) smooth");
    double points[3] = {0.2, 0.5, 0.8};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            double u = points[i], z = points[j];
            double g_rough, g_smooth;
            hdcd_bernstein_tensor_interaction(u, z, m, theta_rough, &g_rough);
            hdcd_bernstein_tensor_interaction(u, z, m, theta_smooth, &g_smooth);
            printf("%8.2f %8.2f %14.4f %14.4f\n", u, z, g_rough, g_smooth);
        }
    }

    free(theta_rough);
    free(theta_smooth);
    return 0;
}
