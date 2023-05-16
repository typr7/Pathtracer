#include "Util.h"

#include <format>
#include <string>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <map>
#include <cstdint>
#include <vector>

#include <glad.h>
#include <glfw3.h>
#include <cuda_runtime_api.h>
#include <Config.h>

#include "Exception.h"

namespace
{

void glfwErrorCallback(int32_t error, const char* msg)
{
    std::cerr << std::format("GLFW Error {}: {}", error, msg) << std::endl;
}

GLFWwindow* initGlfw(const std::string& title, int32_t width, int32_t height)
{
    glfwSetErrorCallback(glfwErrorCallback);
    if (!glfwInit())
        throw std::runtime_error{ EXCEPTION_MSG("failed to initialize GLFW") };

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
    if (window == nullptr)
        throw std::runtime_error{ EXCEPTION_MSG("failed to create GLFW window") };

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    return window;
}

void loadGl()
{
    if (!gladLoadGL())
        throw std::runtime_error{ EXCEPTION_MSG("failed to Load GL") };

    GL_CHECK(glClearColor(0.212f, 0.271f, 0.31f, 1.0f));
    GL_CHECK(glClear(GL_COLOR_BUFFER_BIT));
}

std::map<std::string, std::string> g_code_cache;

} // namespace

std::string getCudaSourcePath(const std::string& filename)
{
    return std::string{ SAMPLES_PTX_DIR } + "/" + filename;
}

std::string getModelPath(const std::string& filename)
{
    return std::string{ SAMPLES_RESOURCE_DIR } + "/model/" + filename;
}

std::string readDataFromFile(const std::string filename)
{
    std::ifstream ifs{ filename, std::ios::binary };
    if (ifs.good()) {
        std::vector<uint8_t> buffer{ std::istreambuf_iterator<char>{ifs}, {} };
        return std::string{ buffer.begin(), buffer.end() };
    } else {
        throw std::runtime_error{ EXCEPTION_MSG("failed to read file") };
    }
}

GLFWwindow* createGlfwWindow(const std::string& title, int32_t width, int32_t height)
{
    GLFWwindow* window = initGlfw(title, width, height);
    loadGl();

    return window;
}

const char* getCompiledCudaCode(const std::string& sample_name, const std::string& source_name, size_t& code_size)
{
    const char* pcode = nullptr;
    std::string filename = sample_name + "_generated_" + source_name + (SAMPLES_INPUT_GENERATE_OPTIXIR ? ".optixir" : ".ptx");
    std::map<std::string, std::string>::iterator it = g_code_cache.find(filename);

    if (it == g_code_cache.end()) {
        std::string file_path = getCudaSourcePath(filename);
        std::string code = readDataFromFile(file_path);

        auto itr = g_code_cache.insert(std::pair{ filename, std::move(code) });

        code_size = itr.first->second.size();
        pcode = itr.first->second.data();
    } else {
        code_size = it->second.size();
        pcode = it->second.data();
    }

    return pcode;
}