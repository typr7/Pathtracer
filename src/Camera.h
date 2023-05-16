#pragma once

#include <cstdint>

#include <vector_types.h>

#include "VectorMath.h"

class Camera
{
public:
    Camera() = default;
    Camera(const float3& position, const float3& target, const float3& up, float vfov, float aspect_ratio, float world_scale);
    ~Camera() noexcept = default;

    bool isValid() const;

    void setPosition(const float3& position) { m_position = position; m_uvw_changed = true; }
    void setTarget(const float3& target) { m_target = target; m_uvw_changed = true; }
    void setUp(const float3& up) { m_up = normalize(up); m_uvw_changed = true; }
    void setFov(float vfov) { m_vfov = vfov; m_uvw_changed = true; }
    void setAspectRatio(float aspect_ratio) { m_aspect_ratio = aspect_ratio; m_uvw_changed = true; }
    void setWorldScale(float world_scale) { m_move_speed = 0.1f * world_scale; }

    float3 getPosition() const noexcept { return m_position; }
    float3 getTarget() const noexcept { return m_target; }
    float3 getUp() const noexcept { return m_up; }
    float getFov() const noexcept { return m_vfov; }
    float getAspectRatio() const noexcept { return m_aspect_ratio; }
    float getWorldScale() const noexcept { return 10.0f * m_move_speed; }
    void getUVW(float3& u, float3& v, float3& w) const noexcept { u = m_ulen * m_u, v = m_vlen * m_v, w = m_wlen * m_w; }

    void zoom(int zoom_length);
    void move(float3 move_offset);
    void rotate(int2 prior_mouse_pos, int2 now_mouse_pos, int2 screen_size);

    void update();
    void computeUVW();

private:
    float3 m_position;
    float3 m_target;
    float3 m_up;
    float  m_vfov;
    float  m_aspect_ratio;
    float  m_move_speed;

    float3 m_u = {};
    float3 m_v = {};
    float3 m_w = {};

    float m_ulen;
    float m_vlen;
    float m_wlen;

    int32_t m_zoom_length = 0;
    float2  m_delta_angle = {};
    float3  m_move_offset = {};

    bool m_zoomed  = false;
    bool m_moved   = false;
    bool m_rotated = false;
    bool m_uvw_changed = false;
};