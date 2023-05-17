#include "Tracer.h"

#include <iostream>
#include <format>
#include <stdexcept>
#include <cstdint>
#include <cassert>
#include <cstring>

#include <glad.h>
#include <glfw3.h>
#include <optix.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <optix_function_table.h>
#include <optix_function_table_definition.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "Exception.h"
#include "Util.h"
#include "CudaBufferView.h"
#include "CudaObjectView.h"
#include "Record.h"
#include "PathTracing.h"

namespace
{

void ensureMinimumSize(int32_t& width, int32_t& height)
{
    if (width <= 0)
        width = 1;
    if (height <= 0)
        height = 1;
}

void optixContextLogCallback(uint32_t level, const char* tag, const char* message, void*)
{
    std::cerr << std::format("[{}][{}]: {}", level, tag, message) << std::endl;
}

struct
{
    OptixIndicesFormat indexFormat(CudaTriangleIndexBufferView::TriangleIndexFormat index_format)
    {
        switch (index_format) {
        case CudaBufferView<TriangleIndexType>::TRIANGLE_INDEX_FORMAT_SHORT3:
        case CudaBufferView<TriangleIndexType>::TRIANGLE_INDEX_FORMAT_USHORT3:
            return OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3;
        case CudaBufferView<TriangleIndexType>::TRIANGLE_INDEX_FORMAT_INT3:
        case CudaBufferView<TriangleIndexType>::TRIANGLE_INDEX_FORMAT_UINT3:
            return OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        default:
            throw std::runtime_error{ std::format("indexFormat(): Unrecognized index format {}", static_cast<uint32_t>(index_format)) };
        }
    }
} g_mapper;

void windowSizeCallback(GLFWwindow* window, int32_t width, int32_t height)
{
    auto state = static_cast<TracerState*>(glfwGetWindowUserPointer(window));

    assert(state != nullptr);

    if (state->minimized)
        return;

    ensureMinimumSize(width, height);

    state->output_size.x  = width;
    state->output_size.y  = height;
    state->window_resized = true;
    state->camera_changed = true;
}

void windowIconifyCallback(GLFWwindow* window, int32_t iconified)
{
    auto state = static_cast<TracerState*>(glfwGetWindowUserPointer(window));

    assert(state != nullptr);

    state->minimized = (iconified > 0);
}

void mouseButtonCallback(GLFWwindow* window, int32_t button, int32_t action, int32_t mods)
{
    auto state = static_cast<TracerState*>(glfwGetWindowUserPointer(window));

    assert(state != nullptr);

    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);

    if (action == GLFW_PRESS && button == GLFW_MOUSE_BUTTON_LEFT) {
        state->prior_mouse_pos = make_int2(static_cast<int>(xpos), static_cast<int>(ypos));
        state->button_pressed  = true;
    } else {
        state->button_pressed = false;
    }
}

void cursorPosCallback(GLFWwindow* window, double xpos, double ypos)
{
    auto state = static_cast<TracerState*>(glfwGetWindowUserPointer(window));

    assert(state != nullptr);

    if (state->button_pressed) {
        if (!glfwGetWindowAttrib(window, GLFW_HOVERED)) {
            state->button_pressed = false;
            return;
        }

        int2 now_mouse_pos = make_int2(static_cast<int32_t>(xpos), static_cast<int32_t>(ypos));
        state->camera.rotate(state->prior_mouse_pos, now_mouse_pos, state->output_size);
        state->prior_mouse_pos = now_mouse_pos;
        state->camera_changed  = true;
    }
}

void scrollCallback(GLFWwindow* window, double xscroll, double yscroll)
{
    auto state = static_cast<TracerState*>(glfwGetWindowUserPointer(window));

    assert(state != nullptr);
    
    state->camera.zoom(static_cast<int>(yscroll));
    state->camera_changed = true;
}

void keyCallback(GLFWwindow* window, int32_t key, int32_t scancode, int32_t action, int32_t mods)
{
    auto state = static_cast<TracerState*>(glfwGetWindowUserPointer(window));

    assert(state != nullptr);

    if(action == GLFW_PRESS || action == GLFW_REPEAT) {
        switch(key) {
        case GLFW_KEY_W:
            state->camera.move(make_float3(0.0f, 0.0f, 1.0f));
            break;
        case GLFW_KEY_S:
            state->camera.move(make_float3(0.0f, 0.0f, -1.0f));
            break;
        case GLFW_KEY_A:
            state->camera.move(make_float3(-1.0f, 0.0f, 0.0f));
            break;
        case GLFW_KEY_D:
            state->camera.move(make_float3(1.0f, 0.0f, 0.0f));
            break;
        case GLFW_KEY_SPACE:
            state->camera.move(make_float3(0.0f, 1.0f, 0.0f));
            break;
        case GLFW_KEY_LEFT_SHIFT:
            state->camera.move(make_float3(0.0f, -1.0f, 0.0f));
            break;
        case GLFW_KEY_P:
            state->gl_switched = true;
            return;
        }
        state->camera_changed = true;
    }
}

}

Tracer::Tracer(int32_t output_width, int32_t output_height)
    : m_gl_context{ "Manet", output_width, output_height }
{
    ensureMinimumSize(output_width, output_height);

    m_state.accum_buffer.resize(sizeof(float4) * output_width * output_height);
    m_state.color_buffer.resize(output_width, output_height);
    m_state.launch_params = LaunchParams{
        .light             = ParallelogramLight{
            .emission = make_float3(100.0f, 100.0f, 50.0f)
            // .emission = make_float3(30.0f)
        },
        .enable_gl         = false,
        .p_rr              = 0.7f,
        .samples_per_pixel = 16,
        .max_tracing_num   = 3,
        .background_color  = make_float3(0.5f, 0.7f, 1.0f)
    };
    m_state.output_size = make_int2(output_width, output_height);

    createOptixContext();
    buildModule();
    buildProgramGroups();
    buildPipeline();
}

Tracer::~Tracer() noexcept
{
    try {
        if (m_state.pipeline)
            OPTIX_CHECK(optixPipelineDestroy(m_state.pipeline));
        if (m_state.raygen_pinhole_pg)
            OPTIX_CHECK(optixProgramGroupDestroy(m_state.raygen_pinhole_pg));
        if (m_state.hit_radiance_pg)
            OPTIX_CHECK(optixProgramGroupDestroy(m_state.hit_radiance_pg));
        if (m_state.hit_occlusion_pg)
            OPTIX_CHECK(optixProgramGroupDestroy(m_state.hit_occlusion_pg));
        if (m_state.miss_radiance_pg)
            OPTIX_CHECK(optixProgramGroupDestroy(m_state.miss_radiance_pg));
        if (m_state.miss_occlusion_pg)
            OPTIX_CHECK(optixProgramGroupDestroy(m_state.miss_occlusion_pg));
        if (m_state.module)
            OPTIX_CHECK(optixModuleDestroy(m_state.module));
        if (m_state.context)
            OPTIX_CHECK(optixDeviceContextDestroy(m_state.context));

        Scene::cleanup(*m_scene);
    } catch (std::exception& e) {
        std::cerr << std::format("Tracer::~Tracer(): {}", e.what()) << std::endl;
    }
}

void Tracer::loadScene(std::shared_ptr<Scene> scene)
{
    m_state.shader_binding_table = {};
    m_state.ias_handle           = 0;
    m_state.buffers.clear();

    assert(scene);
    m_scene = scene;
    buildAccelerationStructures();
    buildShaderBindingTable();

    for (const auto& camera: m_scene->cameras) {
        if (camera.isValid()) {
            m_state.camera = camera;
            break;
        }
    }

    const float3& extent = m_scene->aabb.extent();
    float max_extent = std::max({ extent.x, extent.y, extent.z });


    if (!m_state.camera.isValid()) {
        const float3& center = m_scene->aabb.center();
        m_state.camera = Camera(center + make_float3(0.0f, 0.0f, 4.0f),
                                center,
                                make_float3(0.0f, 1.0f, 0.0f),
                                MANET_PIDIV4,
                                static_cast<float>(m_state.output_size.x)
                                / static_cast<float>(m_state.output_size.y),
                                max_extent);
    }

    /*
    const float3 up = m_state.camera.getUp();
    const float3& center = m_scene->aabb.center();
    ParallelogramLight& light = m_state.launch_params.light;
    light.center = center + 1.2f * max_extent * m_state.camera.getUp();
    light.half_u = 1.5f * extent.x * make_float3(1.0f, 0.0f, 0.0f);
    light.half_v = 1.5f * extent.y * make_float3(0.0f, 0.0f, 1.0f);
    light.normal = -up;
    */

    ParallelogramLight& light = m_state.launch_params.light;
    light.center = make_float3(0.0f, 1.9f, 0.0f);
    light.half_u = make_float3(0.09f, 0.0f, 0.0f);
    light.half_v = make_float3(0.0f, 0.0f, 0.09f);
    light.normal = make_float3(0.0f, -1.0f , 0.0f);
}

void Tracer::start()
{
    m_state.window = m_gl_context.getWindow();
    glfwSetWindowSizeCallback(m_state.window, windowSizeCallback);
    glfwSetWindowIconifyCallback(m_state.window, windowIconifyCallback);
    glfwSetMouseButtonCallback(m_state.window, mouseButtonCallback);
    glfwSetCursorPosCallback(m_state.window, cursorPosCallback);
    glfwSetScrollCallback(m_state.window, scrollCallback);
    glfwSetKeyCallback(m_state.window, keyCallback);
    glfwSetWindowUserPointer(m_state.window, static_cast<void*>(&m_state));

    do {
        glfwPollEvents();
        updateState();
        launch();
        present();
    } while (!glfwWindowShouldClose(m_state.window));
}

void Tracer::createOptixContext()
{
    CUDA_CHECK(cudaFree(nullptr));

    static constexpr CUcontext cuda_context = nullptr;
    OptixDeviceContextOptions options = {
        .logCallbackFunction = optixContextLogCallback,
        .logCallbackLevel    = 4
    };
    OPTIX_CHECK(optixInit());
    OPTIX_CHECK(optixDeviceContextCreate(cuda_context, &options, &m_state.context));
}

void Tracer::buildAccelerationStructures()
{
    // build geometry acceleration structures
    static constexpr uint32_t triangle_input_flags[2] = { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT | OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING,
                                                          OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT };

    OptixTraversableHandle gas_handle = 0;
    CudaDeviceBuffer       gas_buffer;
    {
        std::vector<OptixBuildInput> build_inputs = {};
        for (const auto& mesh: m_scene->meshes) {
            size_t submesh_num = mesh->indices.size();
            assert(mesh->positions.size() == submesh_num
                   && mesh->normals.size() == submesh_num
                   && mesh->texcoords.size() == submesh_num
                   && mesh->colors.size() == submesh_num
                   && mesh->material_indices.size() == submesh_num);
            for (size_t i = 0; i < submesh_num; i++) {
                int32_t material_index = mesh->material_indices[i];
                uint32_t flag_index    = material_index >= 0
                                         ? static_cast<uint32_t>(m_scene->materials[material_index].back_culling)
                                         : 1;
                build_inputs.push_back(OptixBuildInput{
                    .type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES,
                    .triangleArray = OptixBuildInputTriangleArray{
                        .vertexBuffers       = &mesh->positions[i].buffer_ptr,
                        .numVertices         = mesh->positions[i].element_count,
                        .vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3,
                        .vertexStrideInBytes = mesh->positions[i].stride_byte_size,
                        .indexBuffer         = mesh->indices[i].buffer_ptr,
                        .numIndexTriplets    = mesh->indices[i].element_count,
                        .indexFormat         = g_mapper.indexFormat(mesh->indices[i].index_format),
                        .indexStrideInBytes  = mesh->indices[i].stride_byte_size,
                        .preTransform        = 0,
                        .flags               = &triangle_input_flags[flag_index],
                        .numSbtRecords       = 1
                    }
                });
            }
        }
    
        OptixAccelBuildOptions accel_build_options = {
            .buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION,
            .operation  = OPTIX_BUILD_OPERATION_BUILD
        };
        OptixAccelBufferSizes gas_buffer_sizes = {};
        OPTIX_CHECK(optixAccelComputeMemoryUsage(m_state.context,
                                                 &accel_build_options,
                                                 build_inputs.data(),
                                                 static_cast<uint32_t>(build_inputs.size()),
                                                 &gas_buffer_sizes));

        CudaDeviceBuffer temp_buffer(gas_buffer_sizes.tempSizeInBytes);
        CudaDeviceBuffer output_buffer(gas_buffer_sizes.outputSizeInBytes);
        CudaDeviceBuffer compacted_size_buffer(sizeof(size_t));
        OptixAccelEmitDesc emit_desc = {
            .result = compacted_size_buffer.data(),
            .type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE
        };
        OPTIX_CHECK(optixAccelBuild(m_state.context,
                                    nullptr, // stream
                                    &accel_build_options,
                                    build_inputs.data(),
                                    static_cast<uint32_t>(build_inputs.size()),
                                    temp_buffer.data(),
                                    temp_buffer.size(),
                                    output_buffer.data(),
                                    output_buffer.size(),
                                    &gas_handle,
                                    &emit_desc,
                                    1));
        CUDA_SYNC_CHECK();

        size_t compacted_size = 0;
        compacted_size_buffer.download(static_cast<void*>(&compacted_size), sizeof(size_t));

        if (compacted_size < gas_buffer_sizes.outputSizeInBytes) {
            gas_buffer.resize(compacted_size);
            OPTIX_CHECK(optixAccelCompact(m_state.context,
                                          nullptr, // stream
                                          gas_handle,
                                          gas_buffer.data(),
                                          gas_buffer.size(),
                                          &gas_handle));
            CUDA_SYNC_CHECK();
        } else {
            gas_buffer = std::move(output_buffer);
        }
    }

    // build instance acceleration structures
    OptixTraversableHandle ias_handle = 0;
    CudaDeviceBuffer       ias_buffer;
    {
        uint32_t instance_num = static_cast<uint32_t>(m_scene->instances.size());

        std::vector<OptixInstance> optix_instances = {};
        uint32_t sbt_offset = 0;
        for (uint32_t i = 0; i < instance_num; i++) {
            auto instance = m_scene->instances[i];
            optix_instances.push_back(OptixInstance{
                .transform         = {},
                .instanceId        = i,
                .sbtOffset         = sbt_offset,
                .visibilityMask    = 1,
                .flags             = OPTIX_INSTANCE_FLAG_NONE,
                .traversableHandle = gas_handle
            });
            memcpy(optix_instances.back().transform, instance->transform.data(), 12 * sizeof(float));
            sbt_offset += 2 * static_cast<uint32_t>(m_scene->meshes[instance->mesh_index]->indices.size());
        }
    
        CudaDeviceBuffer optix_instance_buffers(static_cast<void*>(optix_instances.data()),
                                                sizeof(OptixInstance) * instance_num);
        OptixBuildInput instance_input = {
            .type          = OPTIX_BUILD_INPUT_TYPE_INSTANCES,
            .instanceArray = OptixBuildInputInstanceArray{
                .instances      = optix_instance_buffers.data(),
                .numInstances   = instance_num,
                .instanceStride = sizeof(OptixInstance)
            }
        };
        OptixAccelBuildOptions accel_build_options = {
            .buildFlags = OPTIX_BUILD_FLAG_NONE,
            .operation  = OPTIX_BUILD_OPERATION_BUILD
        };
        OptixAccelBufferSizes ias_buffer_sizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(m_state.context,
                                                &accel_build_options,
                                                &instance_input,
                                                1, // build input number
                                                &ias_buffer_sizes));

        CudaDeviceBuffer temp_buffer(ias_buffer_sizes.tempSizeInBytes);
        CudaDeviceBuffer output_buffer(ias_buffer_sizes.outputSizeInBytes);
        OPTIX_CHECK(optixAccelBuild(m_state.context,
                                    nullptr, // stream
                                    &accel_build_options,
                                    &instance_input,
                                    1, // build input number
                                    temp_buffer.data(),
                                    temp_buffer.size(),
                                    output_buffer.data(),
                                    output_buffer.size(),
                                    &ias_handle,
                                    nullptr,
                                    0));
        ias_buffer = std::move(output_buffer);
    }

    m_state.ias_handle = ias_handle;
    m_state.buffers.push_back(std::move(gas_buffer));
    m_state.buffers.push_back(std::move(ias_buffer));
}

void Tracer::buildModule()
{
    OptixPayloadType payload_types[2] = {
        {
            .numPayloadValues = static_cast<uint32_t>(std::size(radiance_payload_semantics)),
            .payloadSemantics = radiance_payload_semantics
        },
        {
            .numPayloadValues = static_cast<uint32_t>(std::size(occlusion_payload_semantics)),
            .payloadSemantics = occlusion_payload_semantics
        }
    };

    OptixModuleCompileOptions options = {
#if !defined (NDEBUG)
        .optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0,
        .debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL,
#endif
        .numPayloadTypes = static_cast<uint32_t>(std::size(payload_types)),
        .payloadTypes    = payload_types
    };

    m_state.pipeline_compile_options.usesMotionBlur                   = false;
    m_state.pipeline_compile_options.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS
                                                                        | OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    m_state.pipeline_compile_options.numPayloadValues                 = 0;
    m_state.pipeline_compile_options.numAttributeValues               = 2;
    m_state.pipeline_compile_options.exceptionFlags                   = OPTIX_EXCEPTION_FLAG_NONE;
    m_state.pipeline_compile_options.pipelineLaunchParamsVariableName = "g_launch_params";
    m_state.pipeline_compile_options.usesPrimitiveTypeFlags           = static_cast<uint32_t>(OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);

    size_t code_size = 0;
    const char* code = getCompiledCudaCode("pathtracer", "PathTracing.cu", code_size);

    OPTIX_CHECK(optixModuleCreate(m_state.context,
                                  &options,
                                  &m_state.pipeline_compile_options,
                                  code,
                                  code_size,
                                  nullptr,
                                  nullptr,
                                  &m_state.module));
}

void Tracer::buildProgramGroups()
{
    OptixProgramGroupOptions options = {};
    char log[2048]  = {};
    size_t log_size = 0;

    {
        OptixProgramGroupDesc desc = {
            .kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN,
            .raygen = {
                .module = m_state.module,
                .entryFunctionName = "__raygen__pinhole"
            }
        };
        OPTIX_CHECK(optixProgramGroupCreate(m_state.context,
                                            &desc,
                                            1,
                                            &options,
                                            log,
                                            &log_size,
                                            &m_state.raygen_pinhole_pg));
    }

    if (log_size > 0)
        std::cerr << log << std::endl;

    {
        OptixProgramGroupDesc desc = {
            .kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
            .hitgroup = {
                .moduleCH = m_state.module,
                .entryFunctionNameCH = "__closesthit__radiance"
            }
        };
        OPTIX_CHECK(optixProgramGroupCreate(m_state.context,
                                            &desc,
                                            1,
                                            &options,
                                            log,
                                            &log_size,
                                            &m_state.hit_radiance_pg));
    }

    if (log_size > 0)
        std::cerr << log << std::endl;

    {
        OptixProgramGroupDesc desc = {
            .kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
            .hitgroup = {
                .moduleCH = m_state.module,
                .entryFunctionNameCH = "__closesthit__occlusion"
            }
        };
        OPTIX_CHECK(optixProgramGroupCreate(m_state.context,
                                            &desc,
                                            1,
                                            &options,
                                            log,
                                            &log_size,
                                            &m_state.hit_occlusion_pg));
    }

    if (log_size > 0)
        std::cerr << log << std::endl;

    {
        OptixProgramGroupDesc desc = {
            .kind = OPTIX_PROGRAM_GROUP_KIND_MISS,
            .miss = {
                .module = m_state.module,
                .entryFunctionName = "__miss__radiance"
            }
        };
        OPTIX_CHECK(optixProgramGroupCreate(m_state.context,
                                            &desc,
                                            1,
                                            &options,
                                            log,
                                            &log_size,
                                            &m_state.miss_radiance_pg));
    }

    if (log_size > 0)
        std::cerr << log << std::endl;
    
    {
        OptixProgramGroupDesc desc = {
            .kind = OPTIX_PROGRAM_GROUP_KIND_MISS,
            .miss = {
                .module = m_state.module,
                .entryFunctionName = "__miss__occlusion"
            }
        };
        OPTIX_CHECK(optixProgramGroupCreate(m_state.context,
                                            &desc,
                                            1,
                                            &options,
                                            log,
                                            &log_size,
                                            &m_state.miss_occlusion_pg));
    }

    if (log_size > 0)
        std::cerr << log << std::endl;
}

void Tracer::buildPipeline()
{
    OptixProgramGroup program_groups[] = {
        m_state.raygen_pinhole_pg,
        m_state.hit_radiance_pg,
        m_state.hit_occlusion_pg,
        m_state.miss_radiance_pg,
        m_state.miss_occlusion_pg
    };
    OptixPipelineLinkOptions pipeline_link_options   = { .maxTraceDepth = 2 };
    OPTIX_CHECK(optixPipelineCreate(m_state.context,
                                    &m_state.pipeline_compile_options,
                                    &pipeline_link_options,
                                    program_groups,
                                    static_cast<uint32_t>(std::size(program_groups)),
                                    nullptr,
                                    nullptr,
                                    &m_state.pipeline));
    OptixStackSizes stack_sizes = {};
    for (auto program_group: program_groups)
        OPTIX_CHECK(optixUtilAccumulateStackSizes(program_group, &stack_sizes, m_state.pipeline));

    uint32_t max_trace_depth = 2;
    uint32_t max_cc_depth    = 0;
    uint32_t max_dc_depth    = 0;
    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes,
                                           max_trace_depth,
                                           max_cc_depth,
                                           max_dc_depth,
                                           &direct_callable_stack_size_from_traversal,
                                           &direct_callable_stack_size_from_state,
                                           &continuation_stack_size));
    uint32_t max_traversal_depth = 1;
    OPTIX_CHECK(optixPipelineSetStackSize(m_state.pipeline,
                                          direct_callable_stack_size_from_traversal,
                                          direct_callable_stack_size_from_state,
                                          continuation_stack_size,
                                          max_traversal_depth));
}

void Tracer::buildShaderBindingTable()
{
    auto& sbt = m_state.shader_binding_table;

    {
        EmptyRecord record = {};
        OPTIX_CHECK(optixSbtRecordPackHeader(m_state.raygen_pinhole_pg, static_cast<void*>(&record)));

        CudaDeviceBuffer record_buffer(static_cast<void*>(&record), sizeof(EmptyRecord));
        sbt.raygenRecord = record_buffer.data();
        m_state.buffers.push_back(std::move(record_buffer));
    }

    {
        EmptyRecord records[2] = {};
        OPTIX_CHECK(optixSbtRecordPackHeader(m_state.miss_radiance_pg, static_cast<void*>(&records[0])));
        OPTIX_CHECK(optixSbtRecordPackHeader(m_state.miss_occlusion_pg, static_cast<void*>(&records[1])));

        CudaDeviceBuffer record_buffer(static_cast<void*>(records), sizeof(records));
        sbt.missRecordBase          = record_buffer.data();
        sbt.missRecordStrideInBytes = sizeof(EmptyRecord);
        sbt.missRecordCount         = static_cast<uint32_t>(std::size(records));
        m_state.buffers.push_back(std::move(record_buffer));
    }

    {
        size_t default_material_index = m_scene->materials.size();
        m_scene->materials.push_back(PbrMaterial{});

        m_state.buffers.push_back(CudaDeviceBuffer(m_scene->materials.data(),
                                                   sizeof(PbrMaterial) * m_scene->materials.size()));
        CUdeviceptr material_cuptr = m_state.buffers.back().data();

        std::vector<HitgroupRecord> records = {};
        for (const auto& instance: m_scene->instances) {
            const auto& mesh = m_scene->meshes[instance->mesh_index];
            for (size_t i = 0; i < mesh->indices.size(); i++) {
                int32_t material_index = mesh->material_indices[i];
                HitgroupRecord record  = {
                    .header = {},
                    .data   = HitgroupData{
                        .indices   = mesh->indices[i],
                        .positions = mesh->positions[i],
                        .normals   = mesh->normals[i],
                        .texcoords = mesh->texcoords[i],
                        .colors    = mesh->colors[i],
                        .material  = (material_index >= 0
                                      ? CudaObjectView<PbrMaterial>(material_cuptr + sizeof(PbrMaterial) * material_index)
                                      : CudaObjectView<PbrMaterial>(material_cuptr + sizeof(PbrMaterial) * default_material_index))
                    }
                };

                OPTIX_CHECK(optixSbtRecordPackHeader(m_state.hit_radiance_pg, &record));
                records.push_back(record);

                OPTIX_CHECK(optixSbtRecordPackHeader(m_state.hit_occlusion_pg, &record));
                records.push_back(record);
            }
        }
        CudaDeviceBuffer record_buffer(static_cast<void*>(records.data()), sizeof(HitgroupRecord) * records.size());
        sbt.hitgroupRecordBase          = record_buffer.data();
        sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
        sbt.hitgroupRecordCount         = static_cast<uint32_t>(records.size());
        m_state.buffers.push_back(std::move(record_buffer));
    }
}

void Tracer::updateState()
{
    auto [width, height] = m_state.output_size;
    if (m_state.window_resized) {
        m_state.accum_buffer.resize(sizeof(float4) * width * height);
        m_state.color_buffer.resize(width, height);
        m_state.launch_params.frame.accum_count = 0;
        m_state.window_resized = false;
    }

    if (m_state.camera_changed) {
        auto& [pos, u, v, w] = m_state.launch_params.camera;

        m_state.camera.setAspectRatio(static_cast<float>(width) / static_cast<float>(height));
        m_state.camera.update();

        pos = m_state.camera.getPosition();
        m_state.camera.getUVW(u, v, w);
        m_state.launch_params.frame.accum_count = 0;

        /*
        std::cerr << std::format("u: ({}, {}, {}), v: ({}, {}, {}), w: ({}, {}, {})",
                                 u.x, u.y, u.z,
                                 v.x, v.y, v.z,
                                 w.x, w.y, w.z) << std::endl;
        */

        m_state.camera_changed = false;
    }

    if (m_state.gl_switched) {
        m_state.launch_params.enable_gl         = !m_state.launch_params.enable_gl;
        m_state.launch_params.frame.accum_count = 0;
        m_state.gl_switched = false;
    }
}

void Tracer::launch()
{
    uint32_t pixel_num = static_cast<uint32_t>(m_state.output_size.x * m_state.output_size.y);
    m_state.launch_params.frame.accum_buffer = CudaBufferView<float4>{
        .buffer_ptr       = m_state.accum_buffer.data(),
        .element_count    = pixel_num,
        .stride_byte_size = sizeof(float4)
    };
    m_state.launch_params.frame.color_buffer = CudaBufferView<uchar4>{
        .buffer_ptr       = reinterpret_cast<CUdeviceptr>(m_state.color_buffer.map()),
        .element_count    = pixel_num,
        .stride_byte_size = sizeof(uchar4)
    };
    m_state.launch_params.handle = m_state.ias_handle;
    m_state.launch_params_buffer.upload(static_cast<void*>(&m_state.launch_params), sizeof(LaunchParams));

    OPTIX_CHECK(optixLaunch(m_state.pipeline,
                            0,
                            m_state.launch_params_buffer.data(),
                            sizeof(LaunchParams),
                            &m_state.shader_binding_table,
                            m_state.output_size.x,
                            m_state.output_size.y,
                            1));
    m_state.color_buffer.unmap();

    CUDA_SYNC_CHECK();

    m_state.launch_params.frame.accum_count++;
}

void Tracer::present()
{
    int32_t res_width;
    int32_t res_height;
    glfwGetFramebufferSize(m_state.window, &res_width, &res_height);
    m_gl_display.display(m_state.color_buffer.width(),
                         m_state.color_buffer.height(),
                         res_width,
                         res_height,
                         m_state.color_buffer.getPbo());
    glfwSwapBuffers(m_state.window);
}