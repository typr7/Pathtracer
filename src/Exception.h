#pragma once

#include <string>

#include <optix_types.h>
#include <driver_types.h>

#define EXCEPTION_MSG(msg) exceptionMessageFormat(__FILE__, __LINE__, msg)

#define OPTIX_CHECK(call) optixCheck(call, #call, __FILE__, __LINE__)

#define CUDA_CHECK(call) cudaCheck(call, #call, __FILE__, __LINE__)

#define CUDA_SYNC_CHECK() cudaSyncCheck(__FILE__, __LINE__)

#define GL_CHECK(call)                      \
    {                                       \
        call;                               \
        glCheck(#call, __FILE__, __LINE__); \
    }

std::string exceptionMessageFormat(const std::string& file, int line, const std::string& msg);

void optixCheck(const OptixResult& res, const std::string& call, const std::string& file, int line);

void cudaCheck(const cudaError_t& res, const std::string& call, const std::string& file, int line);

void cudaSyncCheck(const std::string& file, int line);

void glCheck(const std::string& call, const std::string& file, int line);