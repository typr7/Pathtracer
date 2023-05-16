#pragma once

#include <vector_types.h>
#include <optix_device.h>

#include "Record.h"

struct HitResult
{
    float3 intersection;
    float3 normal;
    float4 color;

    struct
    {
        float2 uv;
        float3 t;
        float3 b;
    } texcoord;
};

static __forceinline__ __device__ HitResult getHitResult(const HitgroupData& data)
{
    HitResult result;
    const uint32_t primitive_index = optixGetPrimitiveIndex();
    const float2   barycentrics    = optixGetTriangleBarycentrics();

    const float b1 = barycentrics.x;
    const float b2 = barycentrics.y;
    const float b0 = 1.0f - b1 - b2;

    const uint3 index = data.indices[primitive_index];

    const float3 p0 = data.positions[index.x];
    const float3 p1 = data.positions[index.y];
    const float3 p2 = data.positions[index.z];
    const float3 intersection = b0 * p0 + b1 * p1 + b2 * p2;
    result.intersection = optixTransformPointFromObjectToWorldSpace(intersection);

    float3 normal;
    if (data.normals) {
        const float3 n0 = data.normals[index.x];
        const float3 n1 = data.normals[index.y];
        const float3 n2 = data.normals[index.z];
        normal = b0 * n0 + b1 * n1 + b2 * n2;
    } else {
        normal = cross(p1 - p0, p2 - p0);
    }
    result.normal = normalize(optixTransformNormalFromObjectToWorldSpace(normal));

    result.color = make_float4(1.0f);
    if (data.colors) {
        const float4 c0 = data.colors[index.x];
        const float4 c1 = data.colors[index.y];
        const float4 c2 = data.colors[index.z];
        result.color = b0 * c0 + b1 * c1 + b2 * c2;
    }

    const float3 dp0 = p0 - p2;
    const float3 dp1 = p1 - p2;

    float2 uv;
    float3 t, b;
    if (data.texcoords) {
        const float2 t0 = data.texcoords[index.x];
        const float2 t1 = data.texcoords[index.y];
        const float2 t2 = data.texcoords[index.z];
        
        const float2 dt0 = t0 - t2;
        const float2 dt1 = t1 - t2;
        const float invdet = 1.0f / (dt0.x * dt1.y - dt0.y * dt1.x);

        uv = b0 * t0 + b1 * t1 + b2 * t2;
        t  = invdet * (dt1.y * dp0 - dt0.y * dp1);
        b  = invdet * (dt0.x * dp1 - dt1.x * dp0);
    } else {
        uv = barycentrics;
        t  = -dp0;
        b  = dp1 - dp0;
    }
    result.texcoord = { uv, normalize(t), normalize(b) };

    return result;
}