#include "hdcd/status.h"

const char *hdcd_status_message(hdcd_status_t status) {
    switch (status) {
        case HDCD_OK:
            return "ok";
        case HDCD_ERROR_INVALID_ARGUMENT:
            return "invalid argument";
        case HDCD_ERROR_NOT_CONVERGED:
            return "optimization did not converge";
        case HDCD_ERROR_NUMERICAL:
            return "numerical error (NaN/Inf encountered)";
        case HDCD_ERROR_ALLOCATION:
            return "memory allocation failure";
        case HDCD_ERROR_UNSUPPORTED:
            return "unsupported operation";
        default:
            return "unknown status code";
    }
}
