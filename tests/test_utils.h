#ifndef HDCD_TEST_UTILS_H
#define HDCD_TEST_UTILS_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* Minimal dependency-free test harness. Each test file defines main()
 * and uses these macros; a failing check prints context and exits(1). */

#define HDCD_CHECK(cond) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            exit(1); \
        } \
    } while (0)

#define HDCD_CHECK_NEAR(actual, expected, tol) \
    do { \
        double _a = (actual); \
        double _e = (expected); \
        double _t = (tol); \
        if (fabs(_a - _e) > _t) { \
            fprintf(stderr, "FAIL %s:%d: |%s - %s| = |%.17g - %.17g| = %.17g > tol %.17g\n", \
                    __FILE__, __LINE__, #actual, #expected, _a, _e, fabs(_a - _e), _t); \
            exit(1); \
        } \
    } while (0)

#define HDCD_PASS(name) \
    do { \
        printf("PASS %s\n", (name)); \
    } while (0)

#endif /* HDCD_TEST_UTILS_H */
