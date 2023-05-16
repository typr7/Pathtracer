find_package(OpenGL REQUIRED)

# glfw begin
if(UNIX)
  # On Unix-like systems, shared libraries can use the soname system.
    set(GLFW_LIB_NAME glfw)
else()
    set(GLFW_LIB_NAME glfw3)
endif()

add_subdirectory(glfw)
set_property(TARGET glfw PROPERTY C_STANDARD 99)
# glfw end

# glad begin
add_library(glad SHARED
  KHR/khrplatform.h
  glad/glad.c
  glad/glad.h
)
target_compile_definitions(glad
  PRIVATE GLAD_GLAPI_EXPORT_BUILD
  PUBLIC  GLAD_GLAPI_EXPORT
)
target_include_directories(glad PUBLIC .)
target_link_libraries(glad PUBLIC ${OPENGL_LIBRARIES})
# glad end
