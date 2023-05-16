#pragma once

#include <cstdint>
#include <string>

struct GLFWwindow;

GLFWwindow* createGlfwWindow(const std::string& title, int32_t width, int32_t height);

const char* getCompiledCudaCode(const std::string& sample_name, const std::string& source_name, size_t& code_size);

std::string getCudaSourcePath(const std::string& filename);

std::string readDataFromFile(const std::string filename);

std::string getModelPath(const std::string& filename);