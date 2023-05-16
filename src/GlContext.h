#pragma once

#include <string>

struct GLFWwindow;

class GlContext
{
public:
    GlContext(const std::string& window_title, int32_t output_width, int32_t output_height);
    ~GlContext() noexcept = default;

    GLFWwindow* getWindow() const noexcept { return m_window; }

private:
    GLFWwindow* m_window;
};