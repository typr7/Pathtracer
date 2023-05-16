#include <iostream>
#include <format>
#include <memory>
#include <stdexcept>

#include "Scene.h"
#include "Tracer.h"

int main()
{
    try {
        auto scene = std::make_shared<Scene>();
        Scene::loadFromGltf(*scene, "cornellbox/cornellbox.gltf");
        // Scene::loadFromGltf(*scene, "sponza/sponza.gltf");
        Tracer tracer(800, 600);
        tracer.loadScene(scene);
        tracer.start();
    } catch (std::exception& e) {
        std::cerr << std::format("Caught a exception: {}", e.what()) << std::endl;
    }
    glfwTerminate();

    return 0;
}