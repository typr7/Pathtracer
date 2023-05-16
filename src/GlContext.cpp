#include "GlContext.h"

#include "Util.h"

GlContext::GlContext(const std::string& window_title, int32_t output_width, int32_t output_height)
{
    m_window = createGlfwWindow(window_title, output_width, output_height);
}