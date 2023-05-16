#include "PathTracing.h"

#include <vector_types.h>
#include <optix_device.h>

#include "Manet.h"
#include "random.h"
#include "Helper.h"
#include "Record.h"
#include "HitResult.h"

namespace
{

struct Onb
{
    __forceinline__ __device__ Onb(const float3& normal)
    {
        m_normal = normal;

        if(fabs(m_normal.x) > fabs(m_normal.z))
        {
            m_binormal.x = -m_normal.y;
            m_binormal.y =  m_normal.x;
            m_binormal.z =  0;
        }
        else
        {
            m_binormal.x =  0;
            m_binormal.y = -m_normal.z;
            m_binormal.z =  m_normal.y;
        }

        m_binormal = normalize(m_binormal);
        m_tangent = cross(m_binormal, m_normal);
    }

    __forceinline__ __device__ void inverse_transform(float3& p) const
    {
        p = p.x * m_tangent + p.y * m_binormal + p.z * m_normal;
    }

    float3 m_tangent;
    float3 m_binormal;
    float3 m_normal;
};

__constant__ struct
{
    __forceinline__ __device__ OptixPayloadTypeID constexpr payloadType(PayloadType payload_type)
    {
        switch (payload_type) {
        case PAYLOAD_TYPE_RADIANCE:
            return OPTIX_PAYLOAD_TYPE_ID_0;
        case PAYLOAD_TYPE_OCCLUSION:
            return OPTIX_PAYLOAD_TYPE_ID_1;
        }
    }

} g_mapper;

__forceinline__ __device__ RadiancePayload& getRadiancePayload()
{
    return getPayload<RadiancePayload&>();
}

__forceinline__ __device__ void
traceRadiance(OptixTraversableHandle handle,
              float                  tmin,
              float                  tmax,
              RadiancePayload&       payload)
{
    uint32_t u0, u1;
    packPointer(u0, u1, &payload);

    optixTrace(g_mapper.payloadType(PAYLOAD_TYPE_RADIANCE),
               handle,
               payload.ray_origin,
               payload.ray_direction,
               tmin,
               tmax,
               0.0f,  // ray time
               OptixVisibilityMask{ 255 },    // visibility mask
               OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_CULL_BACK_FACING_TRIANGLES,
               // OPTIX_RAY_FLAG_NONE,
               RAY_TYPE_RADIANCE,
               RAY_TYPE_COUNT,
               RAY_TYPE_RADIANCE,
               u0, u1);
}

__forceinline__ __device__ bool
traceOcclusion(OptixTraversableHandle handle,
               float3                 ray_origin,
               float3                 ray_direction,
               float                  tmin,
               float                  tmax)
{
    uint32_t occluded;
    optixTrace(g_mapper.payloadType(PAYLOAD_TYPE_OCCLUSION),
               handle,
               ray_origin,
               ray_direction,
               tmin,
               tmax,
               0.0f, // ray time
               OptixVisibilityMask{ 255 },    // visibility mask
               OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
               // OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
               RAY_TYPE_OCCLUSION,
               RAY_TYPE_COUNT,
               RAY_TYPE_OCCLUSION,
               occluded);
    return static_cast<bool>(occluded);
}

} // namespace

extern "C" __constant__ LaunchParams g_launch_params;

extern "C" __global__ void __raygen__pinhole()
{
    const uint3    idx               = optixGetLaunchIndex();
    const uint3    dim               = optixGetLaunchDimensions();
    const size_t   pixel_index       = dim.x * idx.y + idx.x;
    const uint32_t accum_count       = g_launch_params.frame.accum_count;
    const uint32_t samples_per_pixel = g_launch_params.samples_per_pixel;
    const uint32_t max_tracing_num   = g_launch_params.max_tracing_num;
    const auto     camera            = g_launch_params.camera;

    uint32_t seed = tea<4>(pixel_index, accum_count);
    float3 result = make_float3(0.0f);

    for (uint32_t i = 0; i < samples_per_pixel; i++) {
        const float2 subpixel_jitter = make_float2(rnd(seed), rnd(seed));
        const float2 st = 2.0f
                          * make_float2((static_cast<float>(idx.x) + subpixel_jitter.x) / static_cast<float>(dim.x),
                                        (static_cast<float>(idx.y) + subpixel_jitter.y) / static_cast<float>(dim.y))
                          - 1.0f;
        RadiancePayload payload = {};
        payload.seed          = seed;
        payload.attenuation   = make_float3(1.0f);
        payload.ray_origin    = camera.position;
        payload.ray_direction = normalize(st.x * camera.u + st.y * camera.v - camera.w);
        payload.seed          = seed;
        payload.depth         = 0;

        do {
            traceRadiance(g_launch_params.handle, 0.001f, 1e15f, payload);
            result += payload.emission;
            result += payload.radiance;
            payload.depth++;
        } while (!payload.done && payload.depth < max_tracing_num);
    }

    float3 result_color = result / static_cast<float>(samples_per_pixel);
    float3 accum_color  = make_float3(g_launch_params.frame.accum_buffer[pixel_index]);
    
    if (accum_count > 0) {
        const float factor = 1.0f / static_cast<float>(accum_count + 1);
        result_color = lerp(accum_color, result_color, factor);
    }

    g_launch_params.frame.accum_buffer[pixel_index] = make_float4(result_color, 1.0f);
    g_launch_params.frame.color_buffer[pixel_index] = make_color(result_color);
}

extern "C" __global__ void __closesthit__radiance()
{
    optixSetPayloadTypes(PAYLOAD_TYPE_RADIANCE);

    const HitgroupData*       data     = reinterpret_cast<HitgroupData*>(optixGetSbtDataPointer());
    const HitResult           result   = getHitResult(*data);
    const PbrMaterial&        material = *(data->material);
    const ParallelogramLight& light    = g_launch_params.light;
    RadiancePayload&          payload  = getRadiancePayload();
    uint32_t                  seed     = payload.seed;

    float3 radiance = make_float3(0.0f);

    float4 base_color = material.base_color * result.color;
    if (material.base_color_texture) {
        const float4 texture_color        = sampleTexture<float4>(material.base_color_texture, result);
        const float3 texture_color_linear = gammaCorrect(make_float3(texture_color), 2.2f); // texture color is stored in sRGB colorspace, add a gamma correction(gamma = 2.2) to it .
        base_color *= make_float4(texture_color_linear, texture_color.w);
    }
    
    if (payload.depth == 0) {
        const float3 emissive_factor = material.emissive_factor;
        float3       emissive_color  = make_float3(1.0f);
        if (material.emissive_texture)
            emissive_color = make_float3(sampleTexture<float4>(material.emissive_texture, result));
        payload.emission = emissive_factor * emissive_color;
    } else {
        payload.emission = make_float3(0.0f);
    }

    // normal
    float3 normal = result.normal;
    if (material.normal_texture) {
        const float4 normal_sampled = 2.0f * sampleTexture<float4>(material.normal_texture, result) - make_float4(1.0f);
        const float2 rotation       = material.normal_texture.texcoord_rotation;
        const float2 tb             = make_float2(normal_sampled.x, normal_sampled.y);
        const float2 tb_trans       = make_float2(dot(tb, make_float2(rotation.y, -rotation.x)),
                                                  dot(tb, make_float2(rotation.x, rotation.y)));
        normal = normalize(tb_trans.x * result.texcoord.t + tb_trans.y * result.texcoord.b + normal_sampled.z * result.normal);
    }

    if (dot(normal, payload.ray_direction) > 0.0f)
        normal = -normal;

    float metallic  = material.metallic;
    float roughness = material.roughness;
    if (material.metallic_roughness_texture) {
        const float4 metallic_roughness = sampleTexture<float4>(material.metallic_roughness_texture, result);
        metallic  *= metallic_roughness.z;
        roughness *= metallic_roughness.y;
    }

    const float  alpha  = roughness * roughness;
    const float3 albedo = make_float3(base_color);
    const float3 F0     = lerp(make_float3(0.04f), albedo, metallic);

    // direct light sample
    {
        const float  s = 2.0f * rnd(seed) - 1.0f;
        const float  t = 2.0f * rnd(seed) - 1.0f;
        const float3 light_sample_pos = light.center + s * light.half_u + t * light.half_v;
        const float3 light_sample_dir = normalize(light_sample_pos - result.intersection);
        const float  distance         = length(light_sample_pos - result.intersection);
        const float3 half_vec   = normalize(light_sample_dir - payload.ray_direction);
        const float  N_dot_L    = dot(normal, light_sample_dir);
        const float  N_dot_V    = dot(normal, -payload.ray_direction);
        const float  N_dot_H    = dot(normal, half_vec);
        const float  V_dot_H    = dot(-payload.ray_direction, half_vec);
        const float  LN_dot_IL  = dot(light.normal, -light_sample_dir);
        
        const float3 F      = schlick(F0, V_dot_H);
        const float  D      = ggxNormal(N_dot_H, alpha);
        const float  G_vis  = vis(N_dot_L, N_dot_V, alpha);
        const float3 ks     = F;
        const float3 kd     = (make_float3(1.0f) - ks) * (1.0f - metallic);
        
        // BRDF
        const float3 f_r = MANET_1DIVPI * kd * albedo + D * G_vis * ks * F;

        if (N_dot_L > 0.0f && LN_dot_IL > 0.0f && distance > 0.00001f) {
            const bool  occluded = traceOcclusion(g_launch_params.handle,
                                                result.intersection,
                                                light_sample_dir,
                                                0.001f,
                                                distance - 0.001f);
            if (!occluded) {
                const float area = 4.0f * length(light.half_u) * length(light.half_v);
                radiance = N_dot_L * LN_dot_IL * area / (distance * distance) * light.emission * f_r;
            }
        }
    }

    payload.radiance = radiance * payload.attenuation;

    // uniform hemisphere sample
    {
        float3 hemisphere_sample_dir;
        cosine_sample_hemisphere(rnd(seed), rnd(seed), hemisphere_sample_dir);
        Onb onb(normal);
        onb.inverse_transform(hemisphere_sample_dir);
        hemisphere_sample_dir = normalize(hemisphere_sample_dir);

        const float3 half_vec = normalize(hemisphere_sample_dir - payload.ray_direction);
        const float  N_dot_L  = dot(normal, hemisphere_sample_dir);
        const float  N_dot_V  = dot(normal, -payload.ray_direction);
        const float  N_dot_H  = dot(normal, half_vec);
        const float  V_dot_H  = dot(-payload.ray_direction, half_vec);

        const float3 F      = schlick(F0, V_dot_H);
        const float  D      = ggxNormal(N_dot_H, alpha);
        const float  G_vis  = vis(N_dot_L, N_dot_V, alpha);
        const float3 ks     = F;
        const float3 kd     = (make_float3(1.0f) - ks) * (1.0f - metallic);

        // BRDF
        const float3 f_r = MANET_1DIVPI * kd * albedo + D * G_vis * ks * F;

        payload.attenuation *= 2.0f * MANET_PI * N_dot_L / g_launch_params.p_rr * f_r;

        payload.ray_origin    = result.intersection;
        payload.ray_direction = hemisphere_sample_dir;
    }

    if (rnd(seed) > g_launch_params.p_rr)
        payload.done = true;

    payload.seed = seed;
}

extern "C" __global__ void __miss__radiance()
{
    optixSetPayloadTypes(PAYLOAD_TYPE_RADIANCE);
    RadiancePayload& payload = getRadiancePayload();

    payload.radiance = make_float3(0.0f); // if depth > 0 and ray missed, then no radiance
    /*
    if (payload.depth == 0) // directly hit the background
        payload.radiance = g_launch_params.background_color;
    */

    payload.emission = make_float3(0.0f);
    payload.done = true;
}

extern "C" __global__ void __closesthit__occlusion()
{
    optixSetPayloadTypes(PAYLOAD_TYPE_OCCLUSION);
    optixSetPayload_0(1);
}

extern "C" __global__ void __miss__occlusion()
{
    optixSetPayloadTypes(PAYLOAD_TYPE_OCCLUSION);
    optixSetPayload_0(0);
}