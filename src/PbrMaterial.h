#pragma once

#include <cstdint>
#include <vector_types.h>
#include <vector_functions.h>

#include <texture_types.h>

#include "ManetMacros.h"

struct PbrMaterial
{
    enum AlphaMode
    {
        ALPHA_MODE_OPAQUE,
        ALPHA_MODE_MASK,
        ALPHA_MODE_BLEND
    };

    struct Texture
    {
        __device__ __forceinline__ operator bool() const { return (texture != 0); }
        
        cudaTextureObject_t texture;

        float2 texcoord_offset;
        float2 texcoord_rotation;
        float2 texcoord_scale;
    };

    bool back_culling = true;

    AlphaMode alpha_mode   = ALPHA_MODE_OPAQUE;
    float     alpha_cutoff = 0.0f;

    float4  base_color       = { 1.0f, 1.0f, 1.0f, 1.0f };
    float3  emissive_factor  = { 0.0f, 0.0f, 0.0f };
    float   metallic         = 1.0f;
    float   roughness        = 1.0f;
    
    Texture base_color_texture         = {};
    Texture emissive_texture           = {};
    Texture metallic_roughness_texture = {};
    Texture normal_texture             = {};
};