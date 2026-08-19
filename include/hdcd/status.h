#ifndef HDCD_STATUS_H
#define HDCD_STATUS_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Explicit status codes. Every numerical routine that can fail must
 * return one of these instead of silently continuing after a NaN/Inf
 * or an invalid argument (spec section 24 and 36, rule 13).
 */
typedef enum hdcd_status {
    HDCD_OK = 0,
    HDCD_ERROR_INVALID_ARGUMENT = 1,
    HDCD_ERROR_NOT_CONVERGED = 2,
    HDCD_ERROR_NUMERICAL = 3,
    HDCD_ERROR_ALLOCATION = 4,
    HDCD_ERROR_UNSUPPORTED = 5
} hdcd_status_t;

/* Human-readable message for a status code. Never returns NULL. */
const char *hdcd_status_message(hdcd_status_t status);

#ifdef __cplusplus
}
#endif

#endif /* HDCD_STATUS_H */
