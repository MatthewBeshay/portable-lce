#include "Device.h"

#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>
#include <vma/vk_mem_alloc.h>

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace plce::vk {

namespace {

constexpr const char* kAppName = "portable-lce";

[[noreturn]] void vk_throw(const char* what, VkResult r) {
    char buf[256];
    std::snprintf(buf, sizeof(buf), "%s failed: VkResult=%d", what, int(r));
    throw std::runtime_error(buf);
}

void vk_check(VkResult r, const char* what) {
    if (r != VK_SUCCESS) vk_throw(what, r);
}

VKAPI_ATTR VkBool32 VKAPI_CALL debug_cb(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT* data, void*) {
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        std::fprintf(stderr, "[vk-validation] %s\n", data->pMessage);
    }
    return VK_FALSE;
}

bool layer_available(const char* name) {
    uint32_t n = 0;
    vkEnumerateInstanceLayerProperties(&n, nullptr);
    std::vector<VkLayerProperties> props(n);
    vkEnumerateInstanceLayerProperties(&n, props.data());
    for (auto& p : props)
        if (std::strcmp(p.layerName, name) == 0) return true;
    return false;
}

}  // namespace

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

Device::Device(SDL_Window* window, bool enable_validation) {
    create_instance(window, enable_validation);
    create_surface(window);
    pick_physical_device();
    create_device();
    create_allocator();
}

Device::~Device() {
    if (device_) vkDeviceWaitIdle(device_);
    if (allocator_) vmaDestroyAllocator(allocator_);
    if (device_)    vkDestroyDevice(device_, nullptr);
    if (surface_)   vkDestroySurfaceKHR(instance_, surface_, nullptr);
    if (debug_messenger_) {
        auto fn = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance_,
                                 "vkDestroyDebugUtilsMessengerEXT"));
        if (fn) fn(instance_, debug_messenger_, nullptr);
    }
    if (instance_) vkDestroyInstance(instance_, nullptr);
}

// ---------------------------------------------------------------------------
// Instance
// ---------------------------------------------------------------------------

void Device::create_instance(SDL_Window* window, bool enable_validation) {
    uint32_t ext_count = 0;
    SDL_Vulkan_GetInstanceExtensions(window, &ext_count, nullptr);
    std::vector<const char*> extensions(ext_count);
    SDL_Vulkan_GetInstanceExtensions(window, &ext_count, extensions.data());

    std::vector<const char*> layers;
    if (enable_validation) {
        if (layer_available("VK_LAYER_KHRONOS_validation")) {
            layers.push_back("VK_LAYER_KHRONOS_validation");
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }
    }
    // Always request debug utils for object naming (no-ops without layers).
    bool has_debug_utils = false;
    for (auto* e : extensions)
        if (std::strcmp(e, VK_EXT_DEBUG_UTILS_EXTENSION_NAME) == 0)
            has_debug_utils = true;
    if (!has_debug_utils)
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName   = kAppName;
    app.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    app.pEngineName        = kAppName;
    app.engineVersion      = VK_MAKE_VERSION(0, 1, 0);
    app.apiVersion         = VK_API_VERSION_1_3;

    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo        = &app;
    ci.enabledExtensionCount   = uint32_t(extensions.size());
    ci.ppEnabledExtensionNames = extensions.data();
    ci.enabledLayerCount       = uint32_t(layers.size());
    ci.ppEnabledLayerNames     = layers.data();

    vk_check(vkCreateInstance(&ci, nullptr, &instance_), "vkCreateInstance");

    if (!layers.empty()) {
        VkDebugUtilsMessengerCreateInfoEXT dci{
            VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
        dci.messageSeverity =
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        dci.messageType =
            VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        dci.pfnUserCallback = debug_cb;
        auto fn = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance_,
                                 "vkCreateDebugUtilsMessengerEXT"));
        if (fn) fn(instance_, &dci, nullptr, &debug_messenger_);
    }
}

// ---------------------------------------------------------------------------
// Surface
// ---------------------------------------------------------------------------

void Device::create_surface(SDL_Window* window) {
    if (!SDL_Vulkan_CreateSurface(window, instance_, &surface_)) {
        throw std::runtime_error(std::string("SDL_Vulkan_CreateSurface: ") +
                                 SDL_GetError());
    }
}

// ---------------------------------------------------------------------------
// Physical device selection
// ---------------------------------------------------------------------------

void Device::pick_physical_device() {
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(instance_, &n, nullptr);
    if (n == 0) throw std::runtime_error("no Vulkan physical devices");
    std::vector<VkPhysicalDevice> devs(n);
    vkEnumeratePhysicalDevices(instance_, &n, devs.data());

    VkPhysicalDevice fallback = VK_NULL_HANDLE;
    uint32_t fallback_family = 0;
    for (auto d : devs) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(d, &props);

        uint32_t qn = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(d, &qn, nullptr);
        std::vector<VkQueueFamilyProperties> qprops(qn);
        vkGetPhysicalDeviceQueueFamilyProperties(d, &qn, qprops.data());

        for (uint32_t i = 0; i < qn; ++i) {
            if (!(qprops[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) continue;
            VkBool32 present = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(d, i, surface_, &present);
            if (!present) continue;

            if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                phys_ = d;
                graphics_family_ = i;
                std::fprintf(stderr, "[vk] device=%s (discrete)\n",
                             props.deviceName);
                return;
            }
            if (!fallback) {
                fallback = d;
                fallback_family = i;
            }
        }
    }

    if (!fallback)
        throw std::runtime_error("no graphics+present queue family");
    phys_ = fallback;
    graphics_family_ = fallback_family;
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(phys_, &props);
    std::fprintf(stderr, "[vk] device=%s\n", props.deviceName);
}

// ---------------------------------------------------------------------------
// Logical device + queue
// ---------------------------------------------------------------------------

void Device::create_device() {
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = graphics_family_;
    qci.queueCount       = 1;
    qci.pQueuePriorities = &prio;

    const char* extensions[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

    VkPhysicalDeviceVulkan13Features v13{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    v13.dynamicRendering = VK_TRUE;
    v13.synchronization2 = VK_TRUE;

    VkPhysicalDeviceFeatures features{};
    VkPhysicalDeviceFeatures supported{};
    vkGetPhysicalDeviceFeatures(phys_, &supported);
    features.wideLines      = supported.wideLines;
    features.depthBiasClamp = supported.depthBiasClamp;

    VkDeviceCreateInfo ci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    ci.pNext                   = &v13;
    ci.queueCreateInfoCount    = 1;
    ci.pQueueCreateInfos       = &qci;
    ci.enabledExtensionCount   = uint32_t(std::size(extensions));
    ci.ppEnabledExtensionNames = extensions;
    ci.pEnabledFeatures        = &features;

    vk_check(vkCreateDevice(phys_, &ci, nullptr, &device_),
             "vkCreateDevice");
    vkGetDeviceQueue(device_, graphics_family_, 0, &graphics_queue_);
}

// ---------------------------------------------------------------------------
// VMA allocator
// ---------------------------------------------------------------------------

void Device::create_allocator() {
    VmaAllocatorCreateInfo ci{};
    ci.physicalDevice = phys_;
    ci.device         = device_;
    ci.instance       = instance_;
    ci.vulkanApiVersion = VK_API_VERSION_1_3;
    vk_check(vmaCreateAllocator(&ci, &allocator_), "vmaCreateAllocator");
}

// ---------------------------------------------------------------------------
// Debug naming
// ---------------------------------------------------------------------------

void Device::set_debug_name(VkObjectType type, uint64_t handle,
                            const char* name) const {
    auto fn = reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(
        vkGetInstanceProcAddr(instance_, "vkSetDebugUtilsObjectNameEXT"));
    if (!fn) return;
    VkDebugUtilsObjectNameInfoEXT ni{
        VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT};
    ni.objectType   = type;
    ni.objectHandle = handle;
    ni.pObjectName  = name;
    fn(device_, &ni);
}

}  // namespace plce::vk
