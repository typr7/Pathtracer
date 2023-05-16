#pragma once

#include <memory>

#include <optix_types.h>

#include "Scene.h"
#include "Manet.h"
#include "Camera.h"
#include "GlContext.h"
#include "GlDisplay.h"
#include "CudaOutputBuffer.h"

struct GLFWwindow;

struct TracerState
{
    LaunchParams             launch_params        = {};
    CudaDeviceBuffer         launch_params_buffer = { sizeof(LaunchParams) };
    CudaDeviceBuffer         accum_buffer         = {};
    CudaOutputBuffer<uchar4> color_buffer         = {};

    OptixDeviceContext          context                  = nullptr;
    OptixModule                 module                   = nullptr;
    OptixPipeline               pipeline                 = nullptr;
    OptixPipelineCompileOptions pipeline_compile_options = {};
    OptixShaderBindingTable     shader_binding_table     = {};

    OptixProgramGroup raygen_pinhole_pg = nullptr;
    OptixProgramGroup hit_radiance_pg   = nullptr;
    OptixProgramGroup hit_occlusion_pg  = nullptr;
    OptixProgramGroup miss_radiance_pg  = nullptr;
    OptixProgramGroup miss_occlusion_pg = nullptr;

    OptixTraversableHandle ias_handle = 0;

    std::vector<CudaDeviceBuffer> buffers = {};

    // window
    int2 output_size = {};
    int2 prior_mouse_pos = {};

    bool button_pressed = false;
    bool window_resized = false;
    bool minimized      = false;

    // camera
    GLFWwindow* window = nullptr;
    Camera camera = {};

    bool camera_changed = true;
};

class Tracer
{
public:
    Tracer(int32_t output_width = 400, int32_t output_height = 300);
    ~Tracer() noexcept;

    void loadScene(std::shared_ptr<Scene> scene);
    
    void start();

private:
    void createOptixContext();
    void buildAccelerationStructures();
    void buildModule();
    void buildProgramGroups();
    void buildPipeline();
    void buildShaderBindingTable();

    void updateState();
    void launch();
    void present();

private:
    GlContext m_gl_context;
    GlDisplay m_gl_display;

    TracerState m_state;

    std::shared_ptr<Scene> m_scene = nullptr;
};