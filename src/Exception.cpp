#include "Exception.h"

#include <format>
#include <stdexcept>
#include <string>

#include <cuda_runtime_api.h>
#include <glad.h>
#include <glfw3.h>
#include <optix.h>

std::string exceptionMessageFormat(const std::string& file, int line, const std::string& msg)
{
    return std::format("[{}, {}]: {}", file, line, msg);
}

void optixCheck(const OptixResult& res, const std::string& call, const std::string& file, int line)
{
    if (res != OPTIX_SUCCESS) {
        std::string msg = std::format("optix call ({}) failed with code {:#010x}", call, static_cast<int>(res));
        throw std::runtime_error(exceptionMessageFormat(file, line, msg));
    }
}

void cudaCheck(const cudaError_t& res, const std::string& call, const std::string& file, int line)
{
    if (res != cudaSuccess) {
        std::string err_name = cudaGetErrorName(res);
        std::string err_msg  = cudaGetErrorString(res);
        std::string msg      = std::format("CUDA call ({}) failed with error: ({}: {})", call, err_name, err_msg);
        throw std::runtime_error(exceptionMessageFormat(file, line, msg));
    }
}

void cudaSyncCheck(const std::string& file, int line)
{
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::string err_name = cudaGetErrorName(err);
        std::string err_msg  = cudaGetErrorString(err);
        std::string msg      = std::format("CUDA sync failed with error: ({}: {})", err_name, err_msg);
        throw std::runtime_error(exceptionMessageFormat(file, line, msg));
    }
}

void glCheck(const std::string& call, const std::string& file, int line)
{
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        std::string msg = std::format("GL call ({}) failed with code {:#010x}", call, static_cast<int>(err));
        throw std::runtime_error(exceptionMessageFormat(file, line, msg));
    }
}