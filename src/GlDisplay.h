#pragma once

#include <cstdint>
#include <string>

class GlDisplay
{
public:
    GlDisplay();
    ~GlDisplay() noexcept;

    GlDisplay(const GlDisplay&)            = delete;
    GlDisplay& operator=(const GlDisplay&) = delete;
    GlDisplay(GlDisplay&&)                 = delete;
    GlDisplay& operator=(GlDisplay&&)      = delete;

    void display(const int32_t screen_width,
                 const int32_t screen_height,
                 const int32_t framebuffer_width,
                 const int32_t framebuffer_height,
                 const uint32_t pbo) const;

private:
    void cleanup() noexcept;

    uint32_t m_render_tex                 = 0;
    uint32_t m_program                    = 0;
    uint32_t m_quad_vertex_buffer         = 0;
    int32_t m_render_tex_uniform_location = -1;

    static const std::string s_vert_source;
    static const std::string s_frag_source;
};