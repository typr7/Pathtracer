#pragma once

#include <cstdint>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "ManetMacros.h"
#include "VectorMath.h"

template <typename ElementType>
struct CudaBufferView
{
    CUdeviceptr buffer_ptr        = 0;
    uint32_t    element_count     = 0;
    uint32_t    stride_byte_size  = 0;

    MANET_DECL bool isValid() const { return (buffer_ptr != 0); }
    MANET_DECL operator bool() const { return isValid(); }

    __forceinline__ __device__ ElementType& operator[](size_t index) const
    {
        return *reinterpret_cast<ElementType*>(buffer_ptr + (stride_byte_size != 0 ? stride_byte_size : sizeof(ElementType)) * index);
    }
};

struct TriangleIndexType {};

template <>
struct CudaBufferView<TriangleIndexType>
{
    enum TriangleIndexFormat
    {
        TRIANGLE_INDEX_FORMAT_SHORT3  = (6 << 8) | 0,
        TRIANGLE_INDEX_FORMAT_USHORT3 = (6 << 8) | 1,
        TRIANGLE_INDEX_FORMAT_INT3    = (12 << 8) | 2,
        TRIANGLE_INDEX_FORMAT_UINT3   = (12 << 8) | 3
    };

    CUdeviceptr         buffer_ptr       = 0;
    uint32_t            element_count    = 0;
    uint32_t            stride_byte_size = 0;
    TriangleIndexFormat index_format     = TRIANGLE_INDEX_FORMAT_UINT3;

    MANET_DECL bool isValid() const { return (buffer_ptr != 0); }
    MANET_DECL operator bool() const { return isValid(); }

    __forceinline__ __device__ uint3 operator[](size_t index) const
    {
        if (index_format == TRIANGLE_INDEX_FORMAT_SHORT3 || index_format == TRIANGLE_INDEX_FORMAT_USHORT3)
            return make_uint3(*reinterpret_cast<ushort3*>(buffer_ptr + (stride_byte_size != 0 ? stride_byte_size : 6) * index));
        else
            return *reinterpret_cast<uint3*>(buffer_ptr + (stride_byte_size != 0 ? stride_byte_size : 12) * index);
    }
};

using CudaTriangleIndexBufferView = CudaBufferView<TriangleIndexType>;