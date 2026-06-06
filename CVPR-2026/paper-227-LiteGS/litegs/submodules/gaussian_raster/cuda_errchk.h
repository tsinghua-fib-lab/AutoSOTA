#pragma once
#include "cuda_runtime.h"
#include <c10/cuda/CUDAException.h>
void cuda_error_check(const char* file, const char* function);
void cuda_error_check_stage(const char* file, const char* function, const char* stage);

//#define CUDA_DEBUG
#ifdef CUDA_DEBUG
    #define CUDA_CHECK_ERRORS cuda_error_check(__FILE__,__FUNCTION__)
#else
    #define CUDA_CHECK_ERRORS
#endif

// Debug-only synchronized stage checks for narrowing down CUDA faults.
// Keep this disabled in normal training/eval builds because each check forces
// a device-wide synchronize and can significantly slow rendering.
//#define LITEGS_ENABLE_RASTER_DEBUG_STAGE_CHECKS
#ifdef LITEGS_ENABLE_RASTER_DEBUG_STAGE_CHECKS
    #define CUDA_CHECK_STAGE(STAGE) cuda_error_check_stage(__FILE__, __FUNCTION__, STAGE)
#else
    #define CUDA_CHECK_STAGE(STAGE)
#endif
