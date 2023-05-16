#pragma once

#include <cmath>

#include <vector_functions.h>
#include <vector_types.h>

#include "ManetMacros.h"

#define TP_PI     3.14159265358f
#define TP_PIDIV2 1.57079632679f
#define TP_PIDIV4 0.78539816339f

// float2

MANET_DECL float2 operator+(float2 a, float2 b) noexcept
{
    return make_float2(a.x + b.x, a.y + b.y);
}

MANET_DECL float2 operator+(float2 a, float f) noexcept
{
    return make_float2(a.x + f, a.y + f);
}

MANET_DECL float2 operator+(float f, float2 a) noexcept
{
    return a + f;
}

MANET_DECL float2& operator+=(float2& a, float2 b) noexcept
{
    a.x += b.x;
    a.y += b.y;

    return a;
}

MANET_DECL float2 operator-(float2 a) noexcept
{
    return make_float2(-a.x, -a.y);
}

MANET_DECL float2 operator-(float2 a, float2 b) noexcept
{
    return make_float2(a.x - b.x, a.y - b.y);
}

MANET_DECL float2 operator-(float2 a, float f) noexcept
{
    return make_float2(a.x - f, a.y - f);
}

MANET_DECL float2& operator-=(float2& a, float2 b) noexcept
{
    a.x -= b.x;
    a.y -= b.y;

    return a;
}

MANET_DECL float2 operator*(float2 a, float2 b) noexcept
{
    return make_float2(a.x * b.x, a.y * b.y);
}

MANET_DECL float2& operator*=(float2& a, float2 b) noexcept
{
    a.x *= b.x;
    a.y *= b.y;

    return a;
}

MANET_DECL float2 operator*(float2 a, float f) noexcept
{
    return make_float2(a.x * f, a.y * f);
}

MANET_DECL float2 operator*(float f, float2 a) noexcept
{
    return a * f;
}

MANET_DECL float2& operator*=(float2& a, float f) noexcept
{
    a.x *= f;
    a.y *= f;

    return a;
}

MANET_DECL float2 operator/(float2 a, float f) noexcept
{
    return make_float2(a.x / f, a.y / f);
}

MANET_DECL float2& operator/=(float2& a, float f) noexcept
{
    a.x /= f;
    a.y /= f;

    return a;
}

// float3

MANET_DECL float3 operator+(const float3& a, const float3& b) noexcept
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

MANET_DECL float3 operator+(const float3& a, float f) noexcept
{
    return make_float3(a.x + f, a.y + f, a.z + f);
}

MANET_DECL float3 operator+(float f, const float3& a) noexcept
{
    return a + f;
}

MANET_DECL float3& operator+=(float3& a, const float3& b) noexcept
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;

    return a;
}

MANET_DECL float3 operator-(const float3& a) noexcept
{
    return make_float3(-a.x, -a.y, -a.z);
}

MANET_DECL float3 operator-(const float3& a, const float3& b) noexcept
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

MANET_DECL float3 operator-(const float3& a, float f) noexcept
{
    return make_float3(a.x - f, a.y - f, a.z - f);
}

MANET_DECL float3& operator-=(float3& a, const float3& b) noexcept
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;

    return a;
}

MANET_DECL float3 operator*(const float3& a, const float3& b) noexcept
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

MANET_DECL float3& operator*=(float3& a, const float3& b) noexcept
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;

    return a;
}

MANET_DECL float3 operator*(const float3& a, float f) noexcept
{
    return make_float3(a.x * f, a.y * f, a.z * f);
}

MANET_DECL float3 operator*(float f, const float3& a) noexcept
{
    return a * f;
}

MANET_DECL float3& operator*=(float3& a, float f) noexcept
{
    a.x *= f;
    a.y *= f;
    a.z *= f;

    return a;
}

MANET_DECL float3 operator/(const float3& a, float f) noexcept
{
    return make_float3(a.x / f, a.y / f, a.z / f);
}

MANET_DECL float3 operator/(const float3& a, const float3& b) noexcept
{
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

MANET_DECL float3& operator/=(float3& a, float f) noexcept
{
    a.x /= f;
    a.y /= f;
    a.z /= f;

    return a;
}

MANET_DECL float3& operator/=(float3& a, const float3& b) noexcept
{
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;

    return a;
}

MANET_DECL float3 make_float3(float f) noexcept
{
    return make_float3(f, f, f);
}

MANET_DECL float3 make_float3(const float4& vec) noexcept
{
    return make_float3(vec.x, vec.y, vec.z);
}

// float4

MANET_DECL float4 operator+(const float4& a, const float4& b) noexcept
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

MANET_DECL float4 operator+(const float4& a, float f) noexcept
{
    return make_float4(a.x + f, a.y + f, a.z + f, a.w + f);
}

MANET_DECL float4 operator+(float f, const float4& a) noexcept
{
    return a + f;
}

MANET_DECL float4& operator+=(float4& a, const float4& b) noexcept
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;

    return a;
}

MANET_DECL float4 operator-(const float4& a) noexcept
{
    return make_float4(-a.x, -a.y, -a.z, -a.w);
}

MANET_DECL float4 operator-(const float4& a, const float4& b) noexcept
{
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

MANET_DECL float4 operator-(const float4& a, float f) noexcept
{
    return make_float4(a.x - f, a.y - f, a.z - f, a.w - f);
}

MANET_DECL float4& operator-=(float4& a, const float4& b) noexcept
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;

    return a;
}

MANET_DECL float4 operator*(const float4& a, const float4& b) noexcept
{
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

MANET_DECL float4& operator*=(float4& a, const float4& b) noexcept
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;

    return a;
}

MANET_DECL float4 operator*(const float4& a, float f) noexcept
{
    return make_float4(a.x * f, a.y * f, a.z * f, a.w * f);
}

MANET_DECL float4 operator*(float f, const float4& a) noexcept
{
    return a * f;
}

MANET_DECL float4& operator*=(float4& a, float f) noexcept
{
    a.x *= f;
    a.y *= f;
    a.z *= f;
    a.w *= f;

    return a;
}

MANET_DECL float4 operator/(const float4& a, float f) noexcept
{
    return make_float4(a.x / f, a.y / f, a.z / f, a.w / f);
}

MANET_DECL float4& operator/=(float4& a, float f) noexcept
{
    a.x /= f;
    a.y /= f;
    a.z /= f;
    a.w /= f;

    return a;
}

MANET_DECL float4 make_float4(float f) noexcept
{
    return make_float4(f, f, f, f);
}

// int3

MANET_DECL int3 operator+(const int3& a, const int3& b) noexcept
{
    return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}

MANET_DECL int3& operator+=(int3& a, const int3& b) noexcept
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    
    return a;
}

// uint3
MANET_DECL uint3 operator+(const uint3& a, const uint3& b) noexcept
{
    return make_uint3(a.x + b.x, a.y + b.y, a.z + b.z);
}

MANET_DECL uint3& operator+=(uint3& a, const uint3& b) noexcept
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    
    return a;
}

MANET_DECL float dot(float2 a, float2 b) noexcept
{
    return a.x * b.x + a.y * b.y;
}

MANET_DECL float dot(const float3& a, const float3& b) noexcept
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

MANET_DECL float lengthSquared(const float3& a) noexcept
{
    return dot(a, a);
}

MANET_DECL float length(const float3& a) noexcept
{
    return std::sqrtf(lengthSquared(a));
}

MANET_DECL float3 cross(const float3& a, const float3& b) noexcept
{
    return make_float3(a.y * b.z - b.y * a.z, b.x * a.z - a.x * b.z, a.x * b.y - b.x * a.y);
}

MANET_DECL float3 normalize(const float3& a) noexcept
{
    return a / length(a);
}

MANET_DECL float3 lerp(const float3& a, const float3& b, const float t)
{
  return a + t * (b - a);
}

MANET_DECL float clamp(float val, float min, float max)
{
    return fminf(max, fmaxf(min, val));
}

MANET_DECL float3 clamp(const float3& v, const float min, const float max)
{
    return make_float3(clamp(v.x, min, max), clamp(v.y, min, max), clamp(v.z, min, max));
}

MANET_DECL float3 clamp(const float3& v, const float3& min, const float3& max)
{
    return make_float3(clamp(v.x, min.x, max.x), clamp(v.y, min.y, max.y), clamp(v.z, min.z, max.z));
}

MANET_DECL float2 min_float2(float2 a, float2 b) noexcept
{
    return make_float2(fminf(a.x, b.x), fminf(a.y, b.y));
}

MANET_DECL float3 min_float3(const float3& a, const float3& b) noexcept
{
    return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

MANET_DECL float4 min_float4(const float4& a, const float4& b) noexcept
{
    return make_float4(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z), fminf(a.w, b.w));
}

MANET_DECL float2 max_float2(float2 a, float2 b) noexcept
{
    return make_float2(fmaxf(a.x, b.x), fmaxf(a.y, b.y));
}

MANET_DECL float3 max_float3(const float3& a, const float3& b) noexcept
{
    return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

MANET_DECL float4 max_float4(const float4& a, const float4& b) noexcept
{
    return make_float4(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z), fmaxf(a.w, b.w));
}

MANET_DECL uint3 make_uint3(const ushort3& v) noexcept
{
    return make_uint3(v.x, v.y, v.z);
}

MANET_DECL float4 make_float4(const float3& xyz, float w) noexcept
{
    return make_float4(xyz.x, xyz.y, xyz.z, w);
}

MANET_DECL uchar4 make_uchar4(const float4& f4) noexcept
{
    return make_uchar4(static_cast<uint8_t>(f4.x),
                       static_cast<uint8_t>(f4.y),
                       static_cast<uint8_t>(f4.z),
                       static_cast<uint8_t>(f4.w));
}