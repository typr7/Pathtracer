#include <iostream>
#include <format>
#include <memory>

#include "Scene.h"
#include "Tracer.h"

void CornellBox(int width, int height)
{
    auto scene = std::make_shared<Scene>();
    Scene::loadFromGltf(*scene, "cornellbox/cornellbox.gltf");
    Tracer tracer(width, height);
    tracer.loadScene(scene);
    tracer.prepareCornellBox();
    tracer.start();
}

void Sponza(int width, int height)
{
    auto scene = std::make_shared<Scene>();
    Scene::loadFromGltf(*scene, "sponza/sponza.gltf");
    Tracer tracer(width, height);
    tracer.loadScene(scene);
    tracer.prepareSponza();
    tracer.start();
}

int main()
{
    try {
        CornellBox(1200, 900);
        // Sponza(1200, 900);
    } catch (std::exception& e) {
        std::cerr << std::format("Caught a exception: {}", e.what()) << std::endl;
    }
    glfwTerminate();

    return 0;
}