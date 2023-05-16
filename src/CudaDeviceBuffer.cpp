#include "CudaDeviceBuffer.h"

#include <cassert>
#include <format>
#include <iostream>

#include <cuda_runtime_api.h>

#include "Exception.h"

namespace
{

void
allocateDeviceMemory(CUdeviceptr& p, size_t byte_size)
{
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&p), byte_size));
}

void
deallocateDeviceMemory(CUdeviceptr p)
{
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(p)));
}

void
memoryCopyHostToDevice(CUdeviceptr dest, const void* src, size_t byte_size)
{
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dest), src, byte_size, cudaMemcpyHostToDevice));
}

void
memoryCopyDeviceToHost(void* dest, CUdeviceptr src, size_t byte_size)
{
    CUDA_CHECK(cudaMemcpy(dest, reinterpret_cast<void*>(src), byte_size, cudaMemcpyDeviceToHost));
}

}  // namespace

CudaDeviceBuffer::CudaDeviceBuffer()
{
    CUDA_CHECK(cudaFree(nullptr));
}

CudaDeviceBuffer::~CudaDeviceBuffer() noexcept
{
    try {
        destory();
    } catch (std::exception& e) {
        std::cerr << std::format("CudaDeviceBuffer destructor caught exception: {}", e.what()) << std::endl;
    }
}

CudaDeviceBuffer::CudaDeviceBuffer(CudaDeviceBuffer&& other)
    : m_data{ other.m_data }
    , m_size{ other.m_size }
{
    CUDA_CHECK(cudaFree(nullptr));
    other.m_data = 0;
    other.m_size = 0;
}

CudaDeviceBuffer&
CudaDeviceBuffer::operator=(CudaDeviceBuffer&& other)
{
    destory();

    m_data       = other.m_data;
    m_size       = other.m_size;
    other.m_data = 0;
    other.m_size = 0;

    return *this;
}

CudaDeviceBuffer::CudaDeviceBuffer(size_t byte_size)
    : m_size{ byte_size }
{
    CUDA_CHECK(cudaFree(nullptr));
    assert(byte_size != 0);

    allocateDeviceMemory(m_data, m_size);
}

CudaDeviceBuffer::CudaDeviceBuffer(const void* src, size_t byte_size)
    : m_size{ byte_size }
{
    CUDA_CHECK(cudaFree(nullptr));
    assert(src != nullptr);
    assert(byte_size != 0);

    allocateDeviceMemory(m_data, m_size);
    memoryCopyHostToDevice(m_data, src, m_size);
}

void
CudaDeviceBuffer::upload(const void* src, size_t byte_size)
{
    assert(m_data != 0);
    assert(src != nullptr);
    assert(byte_size != 0 && byte_size <= m_size);

    memoryCopyHostToDevice(m_data, src, byte_size);
}

size_t
CudaDeviceBuffer::download(void* dest, size_t buffer_size) const
{
    assert(m_data != 0);
    assert(dest != nullptr);
    assert(buffer_size != 0);

    size_t download_size = std::min(buffer_size, m_size);
    memoryCopyDeviceToHost(dest, m_data, download_size);

    return download_size;
}

void
CudaDeviceBuffer::destory()
{
    if (m_data != 0) {
        deallocateDeviceMemory(m_data);
        m_data = 0;
        m_size = 0;
    }
}

void
CudaDeviceBuffer::resize(size_t byte_size)
{
    assert(byte_size != 0);

    destory();
    allocateDeviceMemory(m_data, byte_size);
    m_size = byte_size;
}