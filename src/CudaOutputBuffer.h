#pragma once

#include <iostream>
#include <cstdint>
#include <format>
#include <vector>

#include <cuda.h>
#include <glad.h>
#include <glfw3.h>
#include <cuda_gl_interop.h>

#include "Exception.h"

template <typename PixelFormat>
class CudaOutputBuffer
{
public:
    CudaOutputBuffer() = default;
    ~CudaOutputBuffer() noexcept;

    CudaOutputBuffer(const CudaOutputBuffer&)            = delete;
    CudaOutputBuffer& operator=(const CudaOutputBuffer&) = delete;
    CudaOutputBuffer(CudaOutputBuffer&&)                 = delete;
    CudaOutputBuffer& operator=(CudaOutputBuffer&&)      = delete;

    CudaOutputBuffer(int32_t width, int32_t height);

    void resize(int32_t width, int32_t height);
    void destory();

    PixelFormat* map();
    void unmap();

    GLuint getPbo() { return m_pbo; }
    PixelFormat* getHostPointer();

    int32_t width() const noexcept { return m_width; }
    int32_t height() const noexcept { return m_height; }

private:
    static void ensureMinimumSize(int32_t width, int32_t height)
    {
        if (width <= 0)
            width = 1;
        if (height <= 0)
            height = 1;
    }

private:
    int32_t m_width  = 0;
    int32_t m_height = 0;

    cudaGraphicsResource* m_cuda_gfx_resource = nullptr;
    GLuint m_pbo                              = 0u;

    PixelFormat* m_device_buffer           = nullptr;
    std::vector<PixelFormat> m_host_buffer = {};

    CUstream m_stream        = nullptr;
    int32_t m_cuda_device_id = 0;
};

template <typename PixelFormat>
CudaOutputBuffer<PixelFormat>::CudaOutputBuffer(int32_t width, int32_t height)
{
    ensureMinimumSize(width, height);

    resize(width, height);
}

template <typename PixelFormat>
CudaOutputBuffer<PixelFormat>::~CudaOutputBuffer() noexcept
{
    try {
        destory();
    } catch (std::exception& e) {
        std::cerr << std::format("CudaOutputBuffer destructor caught exception: {}", e.what()) << std::endl;
    }
}

template <typename PixelFormat>
void
CudaOutputBuffer<PixelFormat>::resize(int32_t width, int32_t height)
{
    ensureMinimumSize(width, height);

    m_width  = width;
    m_height = height;

    destory();

    GL_CHECK(glGenBuffers(1, &m_pbo));
    GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, m_pbo));
    GL_CHECK(glBufferData(GL_ARRAY_BUFFER, sizeof(PixelFormat) * m_width * m_height, nullptr, GL_STREAM_DRAW));
    GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, 0));

    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&m_cuda_gfx_resource, m_pbo, cudaGraphicsMapFlagsWriteDiscard));
    m_host_buffer.resize(m_width * m_height);
}

template <typename PixelFormat>
void
CudaOutputBuffer<PixelFormat>::destory()
{
    if (m_pbo == 0u)
        return;

    if (m_device_buffer != nullptr)
        unmap();

    CUDA_CHECK(cudaGraphicsUnregisterResource(m_cuda_gfx_resource));
    GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, 0));
    GL_CHECK(glDeleteBuffers(1, &m_pbo));
    m_host_buffer.clear();
    m_pbo = 0;
}

template <typename PixelFormat>
PixelFormat*
CudaOutputBuffer<PixelFormat>::map()
{
    size_t buffer_size = 0ull;
    CUDA_CHECK(cudaGraphicsMapResources(1, &m_cuda_gfx_resource, m_stream));
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&m_device_buffer),
                                                    &buffer_size,
                                                    m_cuda_gfx_resource));
    return m_device_buffer;
}

template <typename PixelFormat>
void
CudaOutputBuffer<PixelFormat>::unmap()
{
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &m_cuda_gfx_resource, m_stream));
    m_device_buffer = nullptr;
}

template <typename PixelFormat>
PixelFormat*
CudaOutputBuffer<PixelFormat>::getHostPointer()
{
    m_host_buffer.resize(m_width * m_height);

    CUDA_CHECK(cudaMemcpy(static_cast<void*>(m_host_buffer.data()),
                          map(),
                          m_width * m_height * sizeof(PixelFormat),
                          cudaMemcpyDeviceToHost));

    unmap();

    return m_host_buffer.data();
}