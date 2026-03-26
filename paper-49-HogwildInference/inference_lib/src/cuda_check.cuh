#include <stdexcept>
#include <cuda_runtime.h>

#pragma once

/// This exception will be thrown for reported cuda errors
class cuda_error : public std::runtime_error {
public:
    cuda_error(cudaError_t err, const std::string& arg) :
            std::runtime_error(arg), code(err){};

    cudaError_t code;
};

/// Check `status`; if it isn't `cudaSuccess`, throw the corresponding `cuda_error`
inline void cuda_throw_on_error(cudaError_t status, const char* file, int line) {
    if (status != cudaSuccess) {
        std::string msg = std::string("Cuda Error in") + file + ":" + std::to_string(line) + ": " + cudaGetErrorName(status) + ": ";
        msg += cudaGetErrorString(status);
        // make sure we have a clean cuda error state before launching the exception
        // otherwise, if there are calls to the CUDA API in the exception handler,
        // they will return the old error.
        [[maybe_unused]] cudaError_t clear_error = cudaGetLastError();
        throw cuda_error(status, msg);
    }
}

#define CUDA_CHECK_THROW(status) cuda_throw_on_error(status, __FILE__, __LINE__)
#define CUDA_RETURN_ON_ERROR(expr) if(cudaError_t err = expr; err != cudaSuccess) { return err; }