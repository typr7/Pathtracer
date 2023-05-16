#pragma once

#include <cstdint>

#include <vector_types.h>
#include <optix_types.h>

static constexpr uint32_t RAY_TYPE_COUNT = 2;

enum RayType
{
    RAY_TYPE_RADIANCE  = 0,
    RAY_TYPE_OCCLUSION = 1
};

enum PayloadType
{
    PAYLOAD_TYPE_RADIANCE  = static_cast<uint32_t>(OPTIX_PAYLOAD_TYPE_ID_0),
    PAYLOAD_TYPE_OCCLUSION = static_cast<uint32_t>(OPTIX_PAYLOAD_TYPE_ID_1)
};

struct RadiancePayload
{
    float3 attenuation = { 1.0f, 1.0f, 1.0f };
    float3 radiance    = { 0.0f, 0.0f, 0.0f };
    float3 emission    = { 0.0f, 0.0f, 0.0f };
    float3 ray_origin;
    float3 ray_direction;

    uint32_t seed;
    uint32_t depth = 0;
    bool     done  = false;
};

static constexpr uint32_t radiance_payload_semantics[2] = {
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE
    | OPTIX_PAYLOAD_SEMANTICS_CH_READ
    | OPTIX_PAYLOAD_SEMANTICS_MS_READ,
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE
    | OPTIX_PAYLOAD_SEMANTICS_CH_READ
    | OPTIX_PAYLOAD_SEMANTICS_MS_READ
};

static constexpr uint32_t occlusion_payload_semantics[1] = {
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE
};