#include "GlDisplay.h"

#include <format>
#include <stdexcept>

#include <glad.h>

#include "Exception.h"

namespace
{

GLuint
createGlShader(const std::string& source, GLuint shader_type)
{
    GLuint shader = 0;
    try {
        GL_CHECK(shader = glCreateShader(shader_type));

        const GLchar* source_data = source.data();
        GL_CHECK(glShaderSource(shader, 1, &source_data, nullptr));
        GL_CHECK(glCompileShader(shader));

        GLint is_compiled = 0;
        GL_CHECK(glGetShaderiv(shader, GL_COMPILE_STATUS, &is_compiled));
        if (is_compiled == GL_FALSE) {
            GLint max_length = 0;
            GL_CHECK(glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &max_length));

            std::string info_log(max_length, '\0');
            GL_CHECK(glGetShaderInfoLog(shader, max_length, nullptr, info_log.data()));

            throw std::runtime_error(exceptionMessageFormat(__FILE__, __LINE__, std::format("failed to compile shader: {}", info_log)));
        }

        return shader;
    } catch (std::exception& e) {
        if (shader != 0)
            glDeleteShader(shader);
        throw e;
    }
}

GLuint
createGlProgram(const std::string& vert_source, const std::string& frag_source)
{
    GLuint vert_shader = 0;
    GLuint frag_shader = 0;
    GLuint program     = 0;
    try {
        vert_shader = createGlShader(vert_source, GL_VERTEX_SHADER);
        frag_shader = createGlShader(frag_source, GL_FRAGMENT_SHADER);

        GL_CHECK(program = glCreateProgram());
        GL_CHECK(glAttachShader(program, vert_shader));
        GL_CHECK(glAttachShader(program, frag_shader));
        GL_CHECK(glLinkProgram(program));

        GLint is_linked = 0;
        GL_CHECK(glGetProgramiv(program, GL_LINK_STATUS, &is_linked));
        if (is_linked == GL_FALSE) {
            GLint max_length = 0;
            GL_CHECK(glGetProgramiv(program, GL_INFO_LOG_LENGTH, &max_length));

            std::string info_log(max_length, '\0');
            GL_CHECK(glGetProgramInfoLog(program, max_length, nullptr, info_log.data()));

            throw std::runtime_error(exceptionMessageFormat(__FILE__, __LINE__, std::format("failed to link program: {}", info_log)));
        }

        glDeleteShader(vert_shader);
        vert_shader = 0;

        glDeleteShader(frag_shader);
        frag_shader = 0;

        return program;
    } catch (std::exception& e) {
        if (program != 0)
            glDeleteProgram(program);
        if (vert_shader != 0)
            glDeleteShader(vert_shader);
        if (frag_shader != 0)
            glDeleteShader(frag_shader);
        throw e;
    }
}

GLint
getGlUniformLocation(GLuint program, const std::string& uniform_name)
{
    GLint location;
    GL_CHECK(location = glGetUniformLocation(program, uniform_name.c_str()));
    if (location == -1)
        throw std::runtime_error(exceptionMessageFormat(__FILE__, __LINE__, std::format("uniform {} does not exist", uniform_name)));
    return location;
}

}  // namespace

const std::string GlDisplay::s_vert_source = R"(
#version 330 core

layout(location = 0) in vec3 vertexPosition_modelspace;
out vec2 UV;

void main()
{
	gl_Position =  vec4(vertexPosition_modelspace,1);
	UV = (vec2( vertexPosition_modelspace.x, vertexPosition_modelspace.y )+vec2(1,1))/2.0;
}
)";

const std::string GlDisplay::s_frag_source = R"(
#version 330 core

in vec2 UV;
out vec3 color;

uniform sampler2D render_tex;
uniform bool correct_gamma;

void main()
{
    color = texture( render_tex, UV ).xyz;
}
)";

GlDisplay::GlDisplay()
{
    GLuint vertex_array = 0;
    try {
        GL_CHECK(glGenVertexArrays(1, &vertex_array));
        GL_CHECK(glBindVertexArray(vertex_array));

        m_program                     = createGlProgram(s_vert_source, s_frag_source);
        m_render_tex_uniform_location = getGlUniformLocation(m_program, "render_tex");

        GL_CHECK(glGenTextures(1, &m_render_tex));
        GL_CHECK(glBindTexture(GL_TEXTURE_2D, m_render_tex));

        GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
        GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
        GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
        GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));

        static constexpr GLfloat quad_vertex_buffer_data[] = { -1.0f, -1.0f, 0.0f,  1.0f,  -1.0f,
                                                               0.0f,  -1.0f, 1.0f,  0.0f,

                                                               1.0f,  1.0f,  0.0f,  -1.0f, 1.0f,
                                                               0.0f,  1.0f,  -1.0f, 0.0f };

        GL_CHECK(glGenBuffers(1, &m_quad_vertex_buffer));
        GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, m_quad_vertex_buffer));
        GL_CHECK(
            glBufferData(GL_ARRAY_BUFFER, sizeof(quad_vertex_buffer_data), quad_vertex_buffer_data, GL_STATIC_DRAW));
    } catch (std::exception& e) {
        if (vertex_array != 0)
            glDeleteVertexArrays(1, &vertex_array);
        cleanup();

        throw e;
    }
}

GlDisplay::~GlDisplay() noexcept
{
    cleanup();
}

void
GlDisplay::display(const int32_t screen_width,
                   const int32_t screen_height,
                   const int32_t framebuffer_width,
                   const int32_t framebuffer_height,
                   const uint32_t pbo) const
{
    GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, 0));

    GL_CHECK(glViewport(0, 0, framebuffer_width, framebuffer_height));

    GL_CHECK(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));

    GL_CHECK(glUseProgram(m_program));

    GL_CHECK(glActiveTexture(GL_TEXTURE0));
    GL_CHECK(glBindTexture(GL_TEXTURE_2D, m_render_tex));
    GL_CHECK(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo));

    GL_CHECK(glPixelStorei(GL_UNPACK_ALIGNMENT, 4));

    GL_CHECK(
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, screen_width, screen_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr));

    GL_CHECK(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0));
    GL_CHECK(glUniform1i(m_render_tex_uniform_location, 0));

    GL_CHECK(glEnableVertexAttribArray(0));
    GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, m_quad_vertex_buffer));
    GL_CHECK(glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, static_cast<void*>(0)));

    GL_CHECK(glDisable(GL_FRAMEBUFFER_SRGB));

    GL_CHECK(glDrawArrays(GL_TRIANGLES, 0, 6));

    GL_CHECK(glDisableVertexAttribArray(0));
}

void
GlDisplay::cleanup() noexcept
{
    if (m_render_tex != 0) {
        glDeleteTextures(1, &m_render_tex);
        m_render_tex = 0;
    }
    if (m_quad_vertex_buffer != 0) {
        glDeleteBuffers(1, &m_quad_vertex_buffer);
        m_quad_vertex_buffer = 0;
    }
    if (m_program != 0) {
        glDeleteProgram(m_program);
        m_program = 0;
    }
}