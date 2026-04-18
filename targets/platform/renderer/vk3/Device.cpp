#include "Device.h"

#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace plce::vk3 {

namespace {

VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT /*type*/,
    const VkDebugUtilsMessengerCallbackDataEXT* data,
    void* /*user*/) {
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
        std::fprintf(stderr, "[vk] %s\n", data->pMessage);
    return VK_FALSE;
}

bool has_layer(const char* name) {
    auto layers = vk::enumerateInstanceLayerProperties();
    return std::ranges::any_of(layers, [&](const auto& l) {
        return std::strcmp(l.layerName, name) == 0;
    });
}

}  // namespace

Device::Device(const Config& cfg) {
    // --- Dynamic loader bootstrap ---
    // Static dispatch — no init needed (linked against vulkan-loader)

    // --- Instance ---
    uint32_t sdl_ext_n = 0;
    SDL_Vulkan_GetInstanceExtensions(cfg.window, &sdl_ext_n, nullptr);
    std::vector<const char*> exts(sdl_ext_n);
    SDL_Vulkan_GetInstanceExtensions(cfg.window, &sdl_ext_n, exts.data());
    exts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    std::vector<const char*> layers;
    if (cfg.enable_validation && has_layer("VK_LAYER_KHRONOS_validation"))
        layers.push_back("VK_LAYER_KHRONOS_validation");

    vk::ApplicationInfo app_info("portable-lce", 1, "vk3", 1, VK_API_VERSION_1_3);

    vk::InstanceCreateInfo ici({}, &app_info, layers, exts);
    instance_ = vk::raii::Instance(ctx_, ici);

    // --- Debug messenger ---
    if (!layers.empty()) {
        vk::DebugUtilsMessengerCreateInfoEXT dci(
            {},
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,
            vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
                vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
                vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
            debug_callback);
        messenger_ = vk::raii::DebugUtilsMessengerEXT(instance_, dci);
    }

    // --- Surface ---
    VkSurfaceKHR raw_surface = VK_NULL_HANDLE;
    if (!SDL_Vulkan_CreateSurface(cfg.window, *instance_, &raw_surface))
        throw std::runtime_error(std::string("SDL surface: ") + SDL_GetError());
    surface_ = vk::raii::SurfaceKHR(instance_, raw_surface);

    // --- Physical device ---
    auto devices = vk::raii::PhysicalDevices(instance_);
    if (devices.empty()) throw std::runtime_error("no Vulkan devices");

    vk::raii::PhysicalDevice* chosen = nullptr;
    uint32_t chosen_family = 0;
    for (auto& d : devices) {
        auto props = d.getProperties();
        auto qfps  = d.getQueueFamilyProperties();
        for (uint32_t i = 0; i < qfps.size(); ++i) {
            if (!(qfps[i].queueFlags & vk::QueueFlagBits::eGraphics)) continue;
            if (!d.getSurfaceSupportKHR(i, *surface_)) continue;
            if (props.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
                chosen = &d; chosen_family = i;
                std::fprintf(stderr, "[vk3] device=%s\n", props.deviceName.data());
                goto found;
            }
            if (!chosen) { chosen = &d; chosen_family = i; }
        }
    }
    if (!chosen) throw std::runtime_error("no graphics+present queue");
    { auto p = chosen->getProperties();
      std::fprintf(stderr, "[vk3] device=%s\n", p.deviceName.data()); }
found:
    physical_ = std::move(*chosen);
    queue_family_ = chosen_family;

    // --- Logical device ---
    float prio = 1.0f;
    vk::DeviceQueueCreateInfo qci({}, queue_family_, 1, &prio);

    std::array dev_exts = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

    // Vulkan 1.3 features
    vk::PhysicalDeviceVulkan13Features v13;
    v13.dynamicRendering = VK_TRUE;
    v13.synchronization2 = VK_TRUE;

    // Extended dynamic state (for topology, depth bias, etc.)
    vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT eds;
    eds.extendedDynamicState = VK_TRUE;
    eds.pNext = &v13;

    vk::PhysicalDeviceFeatures feats;
    auto supported = physical_.getFeatures();
    feats.wideLines      = supported.wideLines;
    feats.depthBiasClamp = supported.depthBiasClamp;

    vk::DeviceCreateInfo dci({}, qci, {}, dev_exts, &feats, &eds);
    device_ = vk::raii::Device(physical_, dci);
    queue_ = (*device_).getQueue(queue_family_, 0);

    // --- VMA ---
    VmaAllocatorCreateInfo ai{};
    ai.physicalDevice   = *physical_;
    ai.device           = *device_;
    ai.instance         = *instance_;
    ai.vulkanApiVersion = VK_API_VERSION_1_3;
    if (vmaCreateAllocator(&ai, &allocator_) != VK_SUCCESS)
        throw std::runtime_error("vmaCreateAllocator failed");
}

Device::~Device() {
    if (*device_) vkDeviceWaitIdle(*device_);
    if (allocator_) vmaDestroyAllocator(allocator_);
    // vk::raii handles destroy everything else in correct order
}

void Device::name(VkObjectType type, uint64_t obj, const char* label) const {
#ifndef NDEBUG
    if (!*device_) return;
    try {
        vk::DebugUtilsObjectNameInfoEXT ni(type, obj, label);
        (*device_).setDebugUtilsObjectNameEXT(ni);
    } catch (...) {}
#else
    (void)type; (void)obj; (void)label;
#endif
}

}  // namespace plce::vk3
