#include "Device.h"

#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

// Alias to the root vulkan-hpp namespace, disambiguating from `plce::vk`.
namespace vkhpp = ::vk;

namespace plce::vk {

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
    auto layers = vkhpp::enumerateInstanceLayerProperties();
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

    vkhpp::ApplicationInfo app_info("portable-lce", 1, "vk", 1, VK_API_VERSION_1_3);

    vkhpp::InstanceCreateInfo ici({}, &app_info, layers, exts);
    instance_ = vkhpp::raii::Instance(ctx_, ici);

    // --- Debug messenger ---
    if (!layers.empty()) {
        vkhpp::DebugUtilsMessengerCreateInfoEXT dci(
            {},
            vkhpp::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                vkhpp::DebugUtilsMessageSeverityFlagBitsEXT::eError,
            vkhpp::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
                vkhpp::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
                vkhpp::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
            debug_callback);
        messenger_ = vkhpp::raii::DebugUtilsMessengerEXT(instance_, dci);
    }

    // --- Surface ---
    VkSurfaceKHR raw_surface = VK_NULL_HANDLE;
    if (!SDL_Vulkan_CreateSurface(cfg.window, *instance_, &raw_surface))
        throw std::runtime_error(std::string("SDL surface: ") + SDL_GetError());
    surface_ = vkhpp::raii::SurfaceKHR(instance_, raw_surface);

    // --- Physical device ---
    auto devices = vkhpp::raii::PhysicalDevices(instance_);
    if (devices.empty()) throw std::runtime_error("no Vulkan devices");

    vkhpp::raii::PhysicalDevice* chosen = nullptr;
    uint32_t chosen_family = 0;
    for (auto& d : devices) {
        auto props = d.getProperties();
        auto qfps  = d.getQueueFamilyProperties();
        for (uint32_t i = 0; i < qfps.size(); ++i) {
            if (!(qfps[i].queueFlags & vkhpp::QueueFlagBits::eGraphics)) continue;
            if (!d.getSurfaceSupportKHR(i, *surface_)) continue;
            if (props.deviceType == vkhpp::PhysicalDeviceType::eDiscreteGpu) {
                chosen = &d; chosen_family = i;
                std::fprintf(stderr, "[vk] device=%s\n", props.deviceName.data());
                goto found;
            }
            if (!chosen) { chosen = &d; chosen_family = i; }
        }
    }
    if (!chosen) throw std::runtime_error("no graphics+present queue");
    { auto p = chosen->getProperties();
      std::fprintf(stderr, "[vk] device=%s\n", p.deviceName.data()); }
found:
    physical_ = std::move(*chosen);
    queue_family_ = chosen_family;

    // --- Logical device ---
    float prio = 1.0f;
    vkhpp::DeviceQueueCreateInfo qci({}, queue_family_, 1, &prio);

    std::array dev_exts = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

    // Query the physical device's 1.2 feature set so we request only what
    // it actually supports. The bindless path in basic.frag uses
    // nonuniformEXT(tex_id) — if a driver advertises the descriptor-
    // indexing extension but reports shaderSampledImageArrayNonUniform-
    // Indexing = VK_FALSE, enabling it here either fails vkCreateDevice
    // on strict drivers or silently misrenders on lax ones. Refuse up
    // front with a clear message.
    {
        VkPhysicalDeviceVulkan12Features v12_supported{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        VkPhysicalDeviceFeatures2 feats2_query{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        feats2_query.pNext = &v12_supported;
        vkGetPhysicalDeviceFeatures2(*physical_, &feats2_query);
        if (!v12_supported.descriptorIndexing ||
            !v12_supported.runtimeDescriptorArray ||
            !v12_supported.shaderSampledImageArrayNonUniformIndexing ||
            !v12_supported.descriptorBindingSampledImageUpdateAfterBind ||
            !v12_supported.descriptorBindingPartiallyBound ||
            !v12_supported.timelineSemaphore) {
            throw std::runtime_error(
                "vk renderer requires Vulkan 1.2 descriptor-indexing features: "
                "descriptorIndexing, runtimeDescriptorArray, "
                "shaderSampledImageArrayNonUniformIndexing, "
                "descriptorBindingSampledImageUpdateAfterBind, "
                "descriptorBindingPartiallyBound, timelineSemaphore — "
                "physical device reports one or more as unsupported");
        }
    }

    // Vulkan 1.2 features — descriptor indexing for bindless textures,
    // timeline semaphores for upload completion tracking.
    vkhpp::PhysicalDeviceVulkan12Features v12;
    v12.descriptorIndexing                                    = VK_TRUE;
    v12.runtimeDescriptorArray                                = VK_TRUE;
    v12.shaderSampledImageArrayNonUniformIndexing             = VK_TRUE;
    v12.descriptorBindingSampledImageUpdateAfterBind          = VK_TRUE;
    v12.descriptorBindingPartiallyBound                       = VK_TRUE;
    v12.descriptorBindingVariableDescriptorCount              = VK_TRUE;
    v12.descriptorBindingUpdateUnusedWhilePending             = VK_TRUE;
    v12.timelineSemaphore                                     = VK_TRUE;

    // Vulkan 1.3 features. Dynamic rendering + synchronization 2 are used
    // throughout; extendedDynamicState (topology, depth bias, etc.) is core
    // in 1.3 and does not need a separate feature chain.
    vkhpp::PhysicalDeviceVulkan13Features v13;
    v13.dynamicRendering = VK_TRUE;
    v13.synchronization2 = VK_TRUE;
    v13.pNext            = &v12;

    // wideLines is needed to render line primitives with width > 1.0
    // (entity nametag borders at 2.0f, debug graph overlays). Request
    // only if the physical device supports it; otherwise the renderer
    // clamps widths to 1.0.
    vkhpp::PhysicalDeviceFeatures feats;
    auto supported = physical_.getFeatures();
    feats.wideLines      = supported.wideLines;
    wide_lines_enabled_  = supported.wideLines == VK_TRUE;

    // Timestamp support. The timestampPeriod property is nanoseconds per
    // tick; timestampValidBits on the queue family reports how many bits
    // of the 64-bit counter are meaningful (0 means the queue does not
    // support timestamps). FrameContext checks this before emitting
    // vkCmdWriteTimestamp.
    {
        auto props = physical_.getProperties();
        auto qfs   = physical_.getQueueFamilyProperties();
        if (queue_family_ < qfs.size() &&
            qfs[queue_family_].timestampValidBits > 0) {
            timestamp_period_ns_ = props.limits.timestampPeriod;
        }
    }

    vkhpp::DeviceCreateInfo dci({}, qci, {}, dev_exts, &feats, &v13);
    device_ = vkhpp::raii::Device(physical_, dci);
    queue_ = (*device_).getQueue(queue_family_, 0);

    // --- Upload timeline semaphore ---
    // Shared monotonic counter for async texture uploads. Replaces the
    // per-upload VkFence + fence pool in TextureManager; a single
    // vkGetSemaphoreCounterValue read at poll time lets the caller
    // compare against every pending upload's stored value in O(n)
    // memory, O(1) kernel calls.
    {
        vkhpp::SemaphoreTypeCreateInfo ti;
        ti.semaphoreType = vkhpp::SemaphoreType::eTimeline;
        ti.initialValue  = 0;
        vkhpp::SemaphoreCreateInfo sci;
        sci.pNext = &ti;
        upload_timeline_raii_ = vkhpp::raii::Semaphore(device_, sci);
        upload_timeline_      = *upload_timeline_raii_;
    }

    // --- Frame-submission timeline semaphore ---
    // Renderer signals a monotonically-increasing value at the end of
    // every frame's graphics submit; each FrameContext stores the value
    // it last signalled, and StartFrame waits on that value next time
    // the slot comes around. Replaces the per-FrameContext VkFence.
    {
        vkhpp::SemaphoreTypeCreateInfo ti;
        ti.semaphoreType = vkhpp::SemaphoreType::eTimeline;
        ti.initialValue  = 0;
        vkhpp::SemaphoreCreateInfo sci;
        sci.pNext = &ti;
        frame_timeline_raii_ = vkhpp::raii::Semaphore(device_, sci);
        frame_timeline_      = *frame_timeline_raii_;
    }

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

VkResult Device::submit2(uint32_t submit_count, const VkSubmitInfo2* submits,
                         VkFence fence) const {
    std::lock_guard lk(queue_mutex_);
    return vkQueueSubmit2(queue_, submit_count, submits, fence);
}

VkResult Device::present_khr(const VkPresentInfoKHR* present_info) const {
    std::lock_guard lk(queue_mutex_);
    return vkQueuePresentKHR(queue_, present_info);
}

void Device::name(VkObjectType type, uint64_t obj, const char* label) const {
#ifndef NDEBUG
    if (!*device_) return;
    static auto fn = reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(
        vkGetDeviceProcAddr(*device_, "vkSetDebugUtilsObjectNameEXT"));
    if (!fn) return;
    VkDebugUtilsObjectNameInfoEXT ni{VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT};
    ni.objectType   = type;
    ni.objectHandle = obj;
    ni.pObjectName  = label;
    fn(*device_, &ni);
#else
    (void)type; (void)obj; (void)label;
#endif
}

}  // namespace plce::vk
