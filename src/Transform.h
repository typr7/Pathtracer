#pragma once

#include "VectorMath.h"
#include "Matrix.h"

MANET_DECL float3 operator*(const Matrix<4, 4>& matrix, const float3& point)
{
    float4 res = matrix * make_float4(point.x, point.y, point.z, 1.0f);
    return make_float3(res.x / res.w, res.y / res.w, res.z / res.w);
}

MANET_DECL Matrix<4, 4> rotateAroundXAxis(float radian)
{
    float cosr = std::cosf(radian);
    float sinr = std::sinf(radian);
    Matrix<4, 4> ret = {{
        1.0f,  0.0f,  0.0f, 0.0f,
        0.0f,  cosr, -sinr, 0.0f,
        0.0f,  sinr,  cosr, 0.0f,
        0.0f,  0.0f,  0.0f, 1.0f
    }};

    return ret;
}

MANET_DECL Matrix<4, 4> rotateAroundYAxis(float radian)
{
    float cosr = std::cosf(radian);
    float sinr = std::sinf(radian);
    Matrix<4, 4> ret = {{
         cosr, 0.0f,  sinr, 0.0f,
         0.0f, 1.0f,  0.0f, 0.0f,
        -sinr, 0.0f,  cosr, 0.0f,
         0.0f, 0.0f,  0.0f, 1.0f
    }};

    return ret;
}

MANET_DECL Matrix<4, 4> rotateAroundZAxis(float radian)
{
    float cosr = std::cosf(radian);
    float sinr = std::sinf(radian);
    Matrix<4, 4> ret = {{
         cosr, -sinr, 0.0f, 0.0f,
         sinr,  cosr, 0.0f, 0.0f,
         0.0f,  0.0f, 1.0f, 0.0f,
         0.0f,  0.0f, 0.0f, 1.0f
    }};

    return ret;
}

MANET_DECL Matrix<4, 4> rotateAroundAxis(const float3& axis, float radian)
{
    auto [x, y, z] = normalize(axis);
    float xx = x * x;
    float xy = x * y;
    float xz = x * z;
    float yy = y * y;
    float yz = y * z;
    float zz = z * z;

    float cosr  = std::cosf(radian);
    float sinr  = std::sinf(radian);
    float icosr = 1 - cosr;
    Matrix<4, 4> ret = {{
        cosr + xx * icosr, xy * icosr - z * sinr, xz * icosr + y * sinr, 0.0f,
        xy * icosr + z * sinr, cosr + yy * icosr, yz * icosr - x * sinr, 0.0f,
        xz * icosr - y * sinr, yz * icosr + x * sinr, cosr + zz * icosr, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    }};

    return ret;
}

MANET_DECL Matrix<4, 4> identityMatrix()
{
    Matrix<4, 4> ret = {{
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    }};

    return ret;
}

MANET_DECL Matrix<4, 4> translateMatrix(const float3& offset)
{
    Matrix<4, 4> ret = {{
        1.0f, 0.0f, 0.0f, offset.x,
        0.0f, 1.0f, 0.0f, offset.y,
        0.0f, 0.0f, 1.0f, offset.z,
        0.0f, 0.0f, 0.0f,     1.0f
    }};

    return ret;
}

MANET_DECL Matrix<4, 4> rotateMatrix(const float4& quaternion)
{
    float qw = quaternion.w;
    float qx = quaternion.x;
    float qy = quaternion.y;
    float qz = quaternion.z;

    Matrix<4, 4> ret;

    ret[0] = 1.0f - 2.0f * qy * qy - 2.0f * qz * qz;
    ret[1] = 2.0f * qx * qy - 2.0f * qz * qw;
    ret[2] = 2.0f * qx * qz + 2.0f * qy * qw;
    ret[3] = 0.0f;

    ret[4] = 2.0f * qx * qy + 2.0f * qz * qw;
    ret[5] = 1.0f - 2.0f * qx * qx - 2.0f * qz * qz;
    ret[6] = 2.0f * qy * qz - 2.0f * qx * qw;
    ret[7] = 0.0f;

    ret[8] = 2.0f * qx * qz - 2.0f * qy * qw;
    ret[9] = 2.0f * qy * qz + 2.0f * qx * qw;
    ret[10] = 1.0f - 2.0f * qx * qx - 2.0f * qy * qy;
    ret[11] = 0.0f;

    ret[12] = 0.0f;
    ret[13] = 0.0f;
    ret[14] = 0.0f;
    ret[15] = 1.0f;

    return ret;
}

MANET_DECL Matrix<4, 4> scaleMatrix(const float3& scale)
{
    Matrix<4, 4> ret = {{
        scale.x,    0.0f,    0.0f, 0.0f,
           0.0f, scale.y,    0.0f, 0.0f,
           0.0f,    0.0f, scale.z, 0.0f,
           0.0f,    0.0f,    0.0f, 1.0f
    }};

    return ret;
}