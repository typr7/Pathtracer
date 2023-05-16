#pragma once

#include <optix_types.h>

#include "CudaBufferView.h"
#include "CudaObjectView.h"
#include "PbrMaterial.h"

struct Vec2f
{
    MANET_DECL operator float2() const { return make_float2(x, y); }

    float x;
    float y;
};

struct Vec4f
{
    MANET_DECL operator float4() const { return make_float4(x, y, z, w); }

    float x;
    float y;
    float z;
    float w;
};

template <typename T>
struct Record
{
    alignas(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

struct EmptyData {};

struct HitgroupData
{
    CudaTriangleIndexBufferView indices;
    CudaBufferView<float3>      positions;
    CudaBufferView<float3>      normals;
    CudaBufferView<Vec2f>       texcoords;
    CudaBufferView<Vec4f>       colors;
    CudaObjectView<PbrMaterial> material;
};

using EmptyRecord = Record<EmptyData>;
using HitgroupRecord = Record<HitgroupData>;