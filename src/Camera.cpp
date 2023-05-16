#include "Camera.h"

#include <iostream>
#include <format>

#include "Transform.h"

bool Camera::isValid() const
{
    float3 vec = m_target - m_position;
    return (length(vec) > 0.1f) && (fabsf(dot(normalize(vec), m_up)) < 0.9999f);
}

Camera::Camera(const float3& position, const float3& target, const float3& up, float vfov, float aspect_ratio, float world_scale)
    : m_position{ position }
    , m_target{ target }
    , m_up{ normalize(up) }
    , m_vfov{ vfov }
    , m_aspect_ratio{ aspect_ratio }
    , m_move_speed{ 0.1f * world_scale }
{
    computeUVW();
}

void
Camera::computeUVW()
{
    m_w = m_position - m_target;
    m_u = normalize(cross(m_up, m_w));
    m_v = normalize(cross(m_w, m_u));

    m_wlen = length(m_w);
    m_vlen = m_wlen * std::tanf(0.5f * m_vfov);
    m_ulen = m_vlen * m_aspect_ratio;

    m_w = normalize(m_w);
}

void
Camera::zoom(int zoom_length)
{
    if(m_rotated)
        update();

    m_zoom_length += zoom_length;
    m_zoomed = true;
}

void
Camera::move(float3 move_offset)
{
    if(m_rotated)
        update();

    m_move_offset += move_offset;
    m_moved = true;
}

void
Camera::rotate(int2 prior_mouse_pos, int2 now_mouse_pos, int2 screen_size)
{
    if(m_zoomed || m_moved)
        update();

    constexpr auto unitize_pos = [](int pos, int screen) {
        return static_cast<float>(pos) / static_cast<float>(screen);
    };

    float2 pm_pos = make_float2(1.0f - unitize_pos(prior_mouse_pos.x, screen_size.x),
                        unitize_pos(prior_mouse_pos.y, screen_size.y));
    float2 nm_pos = make_float2(1.0f - unitize_pos(now_mouse_pos.x, screen_size.x),
                        unitize_pos(now_mouse_pos.y, screen_size.y));

    float2 dpm = 2.0f * (pm_pos - 0.5f);
    float2 dnm = 2.0f * (nm_pos - 0.5f);

    float x_delta_theta = std::acosf(dnm.x) - std::acosf(dpm.x);
    float y_delta_theta = std::acosf(dnm.y) - std::acosf(dpm.y);

    // std::cerr << std::format("dx: {}, dy: {}", x_delta_theta, y_delta_theta) << std::endl;

    m_delta_angle += make_float2(x_delta_theta, y_delta_theta);
    m_rotated = true;
}

void
Camera::update()
{
    if (m_zoomed) {
        float3 old_position = m_position;
        m_position = m_position - m_move_speed * m_zoom_length * m_w;

        if(!isValid() || dot(normalize(m_target - m_position), normalize(m_target - old_position)) <= 0.0f)
            m_position = old_position;

        m_zoom_length = 0;
        m_zoomed = false;
    }

    if (m_moved) {
        float3 move_offset = m_move_speed * m_move_offset;

        m_position += move_offset.x * m_u + move_offset.y * m_up - move_offset.z * m_w;
        m_target   += move_offset.x * m_u + move_offset.y * m_up - move_offset.z * m_w;

        m_move_offset = { 0.0f, 0.0f, 0.0f };
        m_moved = false;
    }

    if (m_rotated) {
        auto yaw = rotateAroundAxis(m_up, -m_delta_angle.x);

        m_position = (yaw * (m_position - m_target)) + m_target;

        computeUVW();

        auto pitch = rotateAroundAxis(-m_u, -m_delta_angle.y);

        float3 old_position = m_position;
        m_position = (pitch * (m_position - m_target)) + m_target;
        m_uvw_changed = true;

        if (!isValid()) {
            m_position = old_position;
            m_uvw_changed = false;
        }

        m_delta_angle = { 0.0f, 0.0f };
        m_rotated = false;
    }

    if (m_uvw_changed) {
        computeUVW();
        m_uvw_changed = false;
    }
}