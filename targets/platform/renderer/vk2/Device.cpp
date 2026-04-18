#include <vma/vk_mem_alloc.h>

#include "Device.h"

#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace plce::vk2 {

namespace {

[[noreturn]] void die(const char* what, VkResult r) {
    char buf[256];
    std::snprintf(buf, sizeof buf, "%s: VkResult=%d", what, int(r));
    throw std::runtime_error(buf);
}

void check(VkResult r, const char* what) { if (r != VK_SUCCESS) die(what, r); }

VKAPI_ATTR VkBool32 VKAPI_CALL debug_cb(
    VkDebugUtilsMessageSeverityFlagBitsEXT sev,
    VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT* d, void*) {
    if (sev >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
        std::fprintf(stderr, "[vk] %s\n", d->pMessage);
    return VK_FALSE;
}

bool has_layer(const char* name) {
    uint32_t n = 0;
    vkEnumerateInstanceLayerProperties(&n, nullptr);
    std::vector<VkLayerProperties> p(n);
    vkEnumerateInstanceLayerProperties(&n, p.data());
    for (auto& l : p) if (std::strcmp(l.layerName, name) == 0) return true;
    return false;
}

}  // namespace

// ---------------------------------------------------------------------------

Device::Device(const Config& cfg) {
    // --- Instance ---
    uint32_t ext_n = 0;
    SDL_Vulkan_GetInstanceExtensions(cfg.window, &ext_n, nullptr);
    std::vector<const char*> exts(ext_n);
    SDL_Vulkan_GetInstanceExtensions(cfg.window, &ext_n, exts.data());
    exts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    std::vector<const char*> layers;
    if (cfg.enable_validation && has_layer("VK_LAYER_KHRONOS_validation"))
        layers.push_back("VK_LAYER_KHRONOS_validation");

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "portable-lce";
    app.apiVersion       = VK_API_VERSION_1_3;

    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo        = &app;
    ici.enabledExtensionCount   = uint32_t(exts.size());
    ici.ppEnabledExtensionNames = exts.data();
    ici.enabledLayerCount       = uint32_t(layers.size());
    ici.ppEnabledLayerNames     = layers.data();
    check(vkCreateInstance(&ici, nullptr, &instance_), "vkCreateInstance");

    if (!layers.empty()) {
        VkDebugUtilsMessengerCreateInfoEXT dci{
            VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
        dci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                              VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        dci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                          VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                          VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        dci.pfnUserCallback = debug_cb;
        auto fn = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance_, "vkCreateDebugUtilsMessengerEXT"));
        if (fn) fn(instance_, &dci, nullptr, &messenger_);
    }

    pfn_set_name_ = reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(
        vkGetInstanceProcAddr(instance_, "vkSetDebugUtilsObjectNameEXT"));

    // --- Surface ---
    if (!SDL_Vulkan_CreateSurface(cfg.window, instance_, &surface_))
        throw std::runtime_error(std::string("SDL surface: ") + SDL_GetError());

    // --- Physical device ---
    uint32_t dev_n = 0;
    vkEnumeratePhysicalDevices(instance_, &dev_n, nullptr);
    if (!dev_n) throw std::runtime_error("no Vulkan devices");
    std::vector<VkPhysicalDevice> devs(dev_n);
    vkEnumeratePhysicalDevices(instance_, &dev_n, devs.data());

    VkPhysicalDevice fallback = VK_NULL_HANDLE;
    uint32_t fb_family = 0;
    for (auto d : devs) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(d, &props);
        uint32_t qn = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(d, &qn, nullptr);
        std::vector<VkQueueFamilyProperties> qp(qn);
        vkGetPhysicalDeviceQueueFamilyProperties(d, &qn, qp.data());
        for (uint32_t i = 0; i < qn; ++i) {
            if (!(qp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) continue;
            VkBool32 present = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(d, i, surface_, &present);
            if (!present) continue;
            if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                phys_ = d; queue_family_ = i;
                std::fprintf(stderr, "[vk] device=%s\n", props.deviceName);
                goto found;
            }
            if (!fallback) { fallback = d; fb_family = i; }
        }
    }
    if (!fallback) throw std::runtime_error("no graphics+present queue");
    phys_ = fallback; queue_family_ = fb_family;
    {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(phys_, &p);
        std::fprintf(stderr, "[vk] device=%s\n", p.deviceName);
    }
found:

    // --- Logical device ---
    {
        float prio = 1.0f;
        VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci.queueFamilyIndex = queue_family_;
        qci.queueCount       = 1;
        qci.pQueuePriorities = &prio;

        const char* dev_exts[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

        VkPhysicalDeviceVulkan13Features v13{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        v13.dynamicRendering = VK_TRUE;
        v13.synchronization2 = VK_TRUE;

        VkPhysicalDeviceFeatures feats{};
        VkPhysicalDeviceFeatures sup{};
        vkGetPhysicalDeviceFeatures(phys_, &sup);
        feats.wideLines      = sup.wideLines;
        feats.depthBiasClamp = sup.depthBiasClamp;

        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.pNext                   = &v13;
        dci.queueCreateInfoCount    = 1;
        dci.pQueueCreateInfos       = &qci;
        dci.enabledExtensionCount   = uint32_t(std::size(dev_exts));
        dci.ppEnabledExtensionNames = dev_exts;
        dci.pEnabledFeatures        = &feats;
        check(vkCreateDevice(phys_, &dci, nullptr, &device_), "vkCreateDevice");
        vkGetDeviceQueue(device_, queue_family_, 0, &queue_);
    }

    // --- VMA ---
    {
        VmaAllocatorCreateInfo ai{};
        ai.physicalDevice   = phys_;
        ai.device           = device_;
        ai.instance         = instance_;
        ai.vulkanApiVersion = VK_API_VERSION_1_3;
        check(vmaCreateAllocator(&ai, &allocator_), "vmaCreateAllocator");
    }
}

Device::~Device() {
    if (device_)    vkDeviceWaitIdle(device_);
    if (allocator_) vmaDestroyAllocator(allocator_);
    if (device_)    vkDestroyDevice(device_, nullptr);
    if (surface_)   vkDestroySurfaceKHR(instance_, surface_, nullptr);
    if (messenger_) {
        auto fn = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT"));
        if (fn) fn(instance_, messenger_, nullptr);
    }
    if (instance_) vkDestroyInstance(instance_, nullptr);
}

void Device::name(VkObjectType type, uint64_t handle, const char* n) const {
    if (!pfn_set_name_) return;
    VkDebugUtilsObjectNameInfoEXT ni{
        VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT};
    ni.objectType   = type;
    ni.objectHandle = handle;
    ni.pObjectName  = n;
    pfn_set_name_(device_, &ni);
}

}  // namespace plce::vk2
