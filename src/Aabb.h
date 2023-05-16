#pragma once

#include <numeric>

#include <vector_types.h>
#include <vector_functions.h>

#include "VectorMath.h"
#include "Transform.h"

namespace 
{

inline constexpr float inf = std::numeric_limits<float>::infinity();

}

class Aabb
{
public:
    Aabb();
    ~Aabb() noexcept = default;

    Aabb(const Aabb& other) = default;
    Aabb& operator=(const Aabb& other) = default;

    Aabb(const float3& min, const float3& max);

    bool isValid() const noexcept;
    operator bool() const noexcept;

    void invalidate() noexcept;

    void set(const float3& min, const float3& max) noexcept;
    float3 getMin() const noexcept;
    float3 getMax() const noexcept;

    void include(const float3& point) noexcept;
    void include(const Aabb& aabb) noexcept;

    void transform(const Matrix<4, 4>& matrix) noexcept;

    float3 center() const noexcept;

    float3 extent() const noexcept;

private:
    float3 m_min;
    float3 m_max;
};

inline Aabb::Aabb()
    : m_min{ make_float3(inf, inf, inf) }
    , m_max{ make_float3(-inf, -inf, -inf) }
{
}

inline Aabb::Aabb(const float3& min, const float3& max)
    : m_min{ min }, m_max{ max }
{
}

inline bool Aabb::isValid() const noexcept
{
    return (m_min.x <= m_max.x && m_min.y <= m_max.y && m_min.z <= m_max.z);
}

inline Aabb::operator bool() const noexcept
{
    return isValid();
}

inline void Aabb::invalidate() noexcept
{
    set(make_float3(inf, inf, inf), make_float3(-inf, -inf, -inf));
}

inline void Aabb::set(const float3 &min, const float3 &max) noexcept
{
    m_min = min, m_max = max;
}

inline float3 Aabb::getMin() const noexcept
{
    return m_min;
}

inline float3 Aabb::getMax() const noexcept
{
    return m_max;
}

inline void Aabb::include(const float3& point) noexcept
{
    m_min = min_float3(m_min, point);
    m_max = max_float3(m_max, point);
}

inline void Aabb::include(const Aabb& aabb) noexcept
{
    m_min = min_float3(m_min, aabb.m_min);
    m_max = max_float3(m_max, aabb.m_max);
}

inline void Aabb::transform(const Matrix<4, 4>& matrix) noexcept
{
    // get all box vertices
    const float3 b000 = m_min;
    const float3 b001 = make_float3( m_min.x, m_min.y, m_max.z );
    const float3 b010 = make_float3( m_min.x, m_max.y, m_min.z );
    const float3 b011 = make_float3( m_min.x, m_max.y, m_max.z );
    const float3 b100 = make_float3( m_max.x, m_min.y, m_min.z );
    const float3 b101 = make_float3( m_max.x, m_min.y, m_max.z );
    const float3 b110 = make_float3( m_max.x, m_max.y, m_min.z );
    const float3 b111 = m_max;

    invalidate();
    include(matrix * b000);
    include(matrix * b001);
    include(matrix * b010);
    include(matrix * b011);
    include(matrix * b100);
    include(matrix * b101);
    include(matrix * b110);
    include(matrix * b111);
}

inline float3 Aabb::center() const noexcept
{
    return 0.5f * (m_min + m_max);
}

inline float3 Aabb::extent() const noexcept
{
    return m_max - m_min;
}