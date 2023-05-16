#pragma once

#include <cuda.h>

#include "Exception.h"

class CudaDeviceBuffer
{
public:
    CudaDeviceBuffer();
    ~CudaDeviceBuffer() noexcept;

    CudaDeviceBuffer(const CudaDeviceBuffer&)            = delete;
    CudaDeviceBuffer& operator=(const CudaDeviceBuffer&) = delete;

    CudaDeviceBuffer(CudaDeviceBuffer&& other);
    CudaDeviceBuffer& operator=(CudaDeviceBuffer&& other);
    
    CudaDeviceBuffer(size_t byte_size);
    CudaDeviceBuffer(const void* src, size_t byte_size);

    CUdeviceptr data() const noexcept { return m_data; }

    size_t size() const noexcept { return m_size; }

    void resize(size_t byte_size);

    void upload(const void* src, size_t byte_size);

    size_t download(void* dest, size_t buffer_size) const;

    void destory();

private:
    CUdeviceptr m_data = 0;

    size_t m_size = 0;
};