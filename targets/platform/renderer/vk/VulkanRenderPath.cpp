#include "VulkanRenderPath.h"

#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "platform/PlatformTypes.h"

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
        std::fprintf(stderr, "[vk] %s\n", data->pMessage);
    }
    return VK_FALSE;
}

bool layer_available(const char* name) {
    uint32_t n = 0;
    vkEnumerateInstanceLayerProperties(&n, nullptr);
    std::vector<VkLayerProperties> props(n);
    vkEnumerateInstanceLayerProperties(&n, props.data());
    for (auto& p : props) if (std::strcmp(p.layerName, name) == 0) return true;
    return false;
}

}  // namespace

VulkanRenderPath::VulkanRenderPath(SDL_Window* window) : window_(window) {
    int w = 0, h = 0;
    SDL_GetWindowSize(window_, &w, &h);
    fb_.width = uint32_t(w);
    fb_.height = uint32_t(h);
    fb_.aspect = h > 0 ? float(w) / float(h) : 1.0f;

#ifdef NDEBUG
    constexpr bool enable_validation = false;
#else
    constexpr bool enable_validation = true;
#endif

    create_instance(enable_validation);
    create_surface();
    pick_physical_device();
    create_device();
    create_swapchain(uint32_t(w), uint32_t(h));
    create_per_frame();

    std::fprintf(stderr, "[vk] renderer=Vulkan viewport=%ux%u images=%zu\n",
                 swapchain_extent_.width, swapchain_extent_.height,
                 swapchain_views_.size());
}

VulkanRenderPath::~VulkanRenderPath() {
    if (device_) vkDeviceWaitIdle(device_);
    destroy_per_frame();
    destroy_swapchain();
    if (device_)  vkDestroyDevice(device_, nullptr);
    if (surface_) vkDestroySurfaceKHR(instance_, surface_, nullptr);
    if (debug_messenger_) {
        auto fn = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
            instance_, "vkDestroyDebugUtilsMessengerEXT");
        if (fn) fn(instance_, debug_messenger_, nullptr);
    }
    if (instance_) vkDestroyInstance(instance_, nullptr);
}

void VulkanRenderPath::create_instance(bool enable_validation) {
    uint32_t ext_count = 0;
    SDL_Vulkan_GetInstanceExtensions(window_, &ext_count, nullptr);
    std::vector<const char*> extensions(ext_count);
    SDL_Vulkan_GetInstanceExtensions(window_, &ext_count, extensions.data());

    std::vector<const char*> layers;
    if (enable_validation) {
        if (layer_available("VK_LAYER_KHRONOS_validation")) {
            layers.push_back("VK_LAYER_KHRONOS_validation");
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }
    }

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = kAppName;
    app.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    app.pEngineName = kAppName;
    app.engineVersion = VK_MAKE_VERSION(0, 1, 0);
    app.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo = &app;
    ci.enabledExtensionCount = uint32_t(extensions.size());
    ci.ppEnabledExtensionNames = extensions.data();
    ci.enabledLayerCount = uint32_t(layers.size());
    ci.ppEnabledLayerNames = layers.data();

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
        auto fn = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
            instance_, "vkCreateDebugUtilsMessengerEXT");
        if (fn) fn(instance_, &dci, nullptr, &debug_messenger_);
    }
}

void VulkanRenderPath::create_surface() {
    if (!SDL_Vulkan_CreateSurface(window_, instance_, &surface_)) {
        throw std::runtime_error(std::string("SDL_Vulkan_CreateSurface: ") +
                                 SDL_GetError());
    }
}

void VulkanRenderPath::pick_physical_device() {
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(instance_, &n, nullptr);
    if (n == 0) throw std::runtime_error("no Vulkan physical devices");
    std::vector<VkPhysicalDevice> devs(n);
    vkEnumeratePhysicalDevices(instance_, &n, devs.data());

    // Prefer discrete GPU; otherwise first device that has a graphics queue
    // family with present support.
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

    if (!fallback) throw std::runtime_error("no graphics+present queue family");
    phys_ = fallback;
    graphics_family_ = fallback_family;
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(phys_, &props);
    std::fprintf(stderr, "[vk] device=%s\n", props.deviceName);
}

void VulkanRenderPath::create_device() {
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = graphics_family_;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;

    const char* extensions[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

    VkPhysicalDeviceVulkan13Features v13{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    v13.dynamicRendering = VK_TRUE;
    v13.synchronization2 = VK_TRUE;

    VkDeviceCreateInfo ci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    ci.pNext = &v13;
    ci.queueCreateInfoCount = 1;
    ci.pQueueCreateInfos = &qci;
    ci.enabledExtensionCount = std::size(extensions);
    ci.ppEnabledExtensionNames = extensions;

    vk_check(vkCreateDevice(phys_, &ci, nullptr, &device_), "vkCreateDevice");
    vkGetDeviceQueue(device_, graphics_family_, 0, &graphics_queue_);
}

void VulkanRenderPath::create_swapchain(uint32_t width, uint32_t height) {
    VkSurfaceCapabilitiesKHR caps{};
    vk_check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(phys_, surface_, &caps),
             "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");

    uint32_t fmt_count = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(phys_, surface_, &fmt_count, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(fmt_count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(phys_, surface_, &fmt_count,
                                         formats.data());

    VkSurfaceFormatKHR chosen = formats.front();
    for (auto& f : formats) {
        if (f.format == VK_FORMAT_B8G8R8A8_UNORM &&
            f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            chosen = f;
            break;
        }
    }
    swapchain_format_ = chosen.format;

    VkExtent2D extent{width, height};
    if (caps.currentExtent.width != UINT32_MAX) {
        extent = caps.currentExtent;
    }
    extent.width = std::clamp(extent.width, caps.minImageExtent.width,
                              caps.maxImageExtent.width);
    extent.height = std::clamp(extent.height, caps.minImageExtent.height,
                               caps.maxImageExtent.height);
    swapchain_extent_ = extent;

    uint32_t image_count = caps.minImageCount + 1;
    if (caps.maxImageCount > 0 && image_count > caps.maxImageCount) {
        image_count = caps.maxImageCount;
    }

    VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    ci.surface = surface_;
    ci.minImageCount = image_count;
    ci.imageFormat = chosen.format;
    ci.imageColorSpace = chosen.colorSpace;
    ci.imageExtent = extent;
    ci.imageArrayLayers = 1;
    ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                    VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.preTransform = caps.currentTransform;
    ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    ci.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    ci.clipped = VK_TRUE;

    vk_check(vkCreateSwapchainKHR(device_, &ci, nullptr, &swapchain_),
             "vkCreateSwapchainKHR");

    uint32_t img_count = 0;
    vkGetSwapchainImagesKHR(device_, swapchain_, &img_count, nullptr);
    swapchain_images_.resize(img_count);
    vkGetSwapchainImagesKHR(device_, swapchain_, &img_count,
                            swapchain_images_.data());

    swapchain_views_.resize(img_count);
    for (uint32_t i = 0; i < img_count; ++i) {
        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image = swapchain_images_[i];
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = swapchain_format_;
        vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vci.subresourceRange.levelCount = 1;
        vci.subresourceRange.layerCount = 1;
        vk_check(vkCreateImageView(device_, &vci, nullptr,
                                   &swapchain_views_[i]),
                 "vkCreateImageView");
    }

    fb_.width = swapchain_extent_.width;
    fb_.height = swapchain_extent_.height;
    fb_.aspect = swapchain_extent_.height > 0
                     ? float(swapchain_extent_.width) /
                           float(swapchain_extent_.height)
                     : 1.0f;
}

void VulkanRenderPath::destroy_swapchain() {
    for (auto v : swapchain_views_) {
        if (v) vkDestroyImageView(device_, v, nullptr);
    }
    swapchain_views_.clear();
    swapchain_images_.clear();
    if (swapchain_) {
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        swapchain_ = VK_NULL_HANDLE;
    }
}

void VulkanRenderPath::create_per_frame() {
    for (auto& f : frames_) {
        VkCommandPoolCreateInfo pci{
            VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pci.queueFamilyIndex = graphics_family_;
        vk_check(vkCreateCommandPool(device_, &pci, nullptr, &f.pool),
                 "vkCreateCommandPool");

        VkCommandBufferAllocateInfo ai{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = f.pool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        vk_check(vkAllocateCommandBuffers(device_, &ai, &f.cmd),
                 "vkAllocateCommandBuffers");

        VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        vk_check(vkCreateSemaphore(device_, &sci, nullptr, &f.image_acquired),
                 "vkCreateSemaphore");
        vk_check(vkCreateSemaphore(device_, &sci, nullptr, &f.render_done),
                 "vkCreateSemaphore");

        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        vk_check(vkCreateFence(device_, &fci, nullptr, &f.in_flight),
                 "vkCreateFence");
    }
}

void VulkanRenderPath::destroy_per_frame() {
    for (auto& f : frames_) {
        if (f.in_flight)      vkDestroyFence(device_, f.in_flight, nullptr);
        if (f.image_acquired) vkDestroySemaphore(device_, f.image_acquired,
                                                 nullptr);
        if (f.render_done)    vkDestroySemaphore(device_, f.render_done,
                                                 nullptr);
        if (f.pool)           vkDestroyCommandPool(device_, f.pool, nullptr);
        f = {};
    }
}

// ---------------------------------------------------------------------------
// Frame lifecycle
// ---------------------------------------------------------------------------

void VulkanRenderPath::StartFrame() {
    if (frame_active_) return;

    PerFrame& f = frames_[frame_index_];
    vkWaitForFences(device_, 1, &f.in_flight, VK_TRUE, UINT64_MAX);

    VkResult acq = vkAcquireNextImageKHR(device_, swapchain_, UINT64_MAX,
                                         f.image_acquired, VK_NULL_HANDLE,
                                         &acquired_image_);
    if (acq == VK_ERROR_OUT_OF_DATE_KHR) {
        int w = 0, h = 0;
        SDL_GetWindowSize(window_, &w, &h);
        resize(uint32_t(w), uint32_t(h));
        return;
    }
    if (acq != VK_SUCCESS && acq != VK_SUBOPTIMAL_KHR) {
        vk_throw("vkAcquireNextImageKHR", acq);
    }

    vkResetFences(device_, 1, &f.in_flight);
    vkResetCommandBuffer(f.cmd, 0);

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(f.cmd, &bi);

    // UNDEFINED -> TRANSFER_DST_OPTIMAL for the clear.
    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    b.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
    b.srcAccessMask = 0;
    b.dstStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
    b.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b.image = swapchain_images_[acquired_image_];
    b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    b.subresourceRange.levelCount = 1;
    b.subresourceRange.layerCount = 1;
    VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dep.imageMemoryBarrierCount = 1;
    dep.pImageMemoryBarriers = &b;
    vkCmdPipelineBarrier2(f.cmd, &dep);

    VkClearColorValue clear{};
    clear.float32[0] = clear_color_[0];
    clear.float32[1] = clear_color_[1];
    clear.float32[2] = clear_color_[2];
    clear.float32[3] = clear_color_[3];
    VkImageSubresourceRange range{};
    range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    range.levelCount = 1;
    range.layerCount = 1;
    vkCmdClearColorImage(f.cmd, swapchain_images_[acquired_image_],
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear, 1,
                         &range);

    frame_active_ = true;
}

void VulkanRenderPath::Present() {
    if (!frame_active_) return;
    PerFrame& f = frames_[frame_index_];

    // TRANSFER_DST_OPTIMAL -> PRESENT_SRC_KHR
    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    b.srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
    b.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    b.dstStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
    b.dstAccessMask = 0;
    b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    b.image = swapchain_images_[acquired_image_];
    b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    b.subresourceRange.levelCount = 1;
    b.subresourceRange.layerCount = 1;
    VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dep.imageMemoryBarrierCount = 1;
    dep.pImageMemoryBarriers = &b;
    vkCmdPipelineBarrier2(f.cmd, &dep);

    vkEndCommandBuffer(f.cmd);

    VkSemaphoreSubmitInfo wait{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    wait.semaphore = f.image_acquired;
    wait.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSemaphoreSubmitInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    signal.semaphore = f.render_done;
    signal.stageMask = VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT;
    VkCommandBufferSubmitInfo cmd_si{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    cmd_si.commandBuffer = f.cmd;

    VkSubmitInfo2 si{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    si.waitSemaphoreInfoCount = 1;
    si.pWaitSemaphoreInfos = &wait;
    si.commandBufferInfoCount = 1;
    si.pCommandBufferInfos = &cmd_si;
    si.signalSemaphoreInfoCount = 1;
    si.pSignalSemaphoreInfos = &signal;
    vk_check(vkQueueSubmit2(graphics_queue_, 1, &si, f.in_flight),
             "vkQueueSubmit2");

    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = &f.render_done;
    pi.swapchainCount = 1;
    pi.pSwapchains = &swapchain_;
    pi.pImageIndices = &acquired_image_;
    VkResult pr = vkQueuePresentKHR(graphics_queue_, &pi);
    if (pr == VK_ERROR_OUT_OF_DATE_KHR || pr == VK_SUBOPTIMAL_KHR) {
        int w = 0, h = 0;
        SDL_GetWindowSize(window_, &w, &h);
        resize(uint32_t(w), uint32_t(h));
    } else if (pr != VK_SUCCESS) {
        vk_throw("vkQueuePresentKHR", pr);
    }

    frame_active_ = false;
    frame_index_ = (frame_index_ + 1) % kFramesInFlight;
}

void VulkanRenderPath::Clear(int) {
    // Already done in StartFrame for Phase 2; will become per-view in
    // later phases.
}

void VulkanRenderPath::SetClearColour(const float rgba[4]) {
    clear_color_[0] = rgba[0];
    clear_color_[1] = rgba[1];
    clear_color_[2] = rgba[2];
    clear_color_[3] = rgba[3];
}

void VulkanRenderPath::render_frame(const rp::FrameDesc&) {
    if (!frame_active_) StartFrame();
    Present();
}

void VulkanRenderPath::resize(uint32_t w, uint32_t h) {
    if (w == 0 || h == 0) return;
    vkDeviceWaitIdle(device_);
    destroy_swapchain();
    create_swapchain(w, h);
    frame_active_ = false;
}

void VulkanRenderPath::GetFramebufferSize(int& w, int& h) {
    w = int(fb_.width);
    h = int(fb_.height);
}

void VulkanRenderPath::SetWindowSize(int w, int h) { resize(uint32_t(w), uint32_t(h)); }
void VulkanRenderPath::SetFullscreen(bool) {}
void VulkanRenderPath::Close() { should_close_ = true; }
bool VulkanRenderPath::ShouldClose() { return should_close_; }
const rp::FrameFramebuffer& VulkanRenderPath::framebuffer() const { return fb_; }
bool VulkanRenderPath::IsWidescreen() { return fb_.is_widescreen; }
bool VulkanRenderPath::IsHiDef()      { return fb_.is_hi_def; }

const float* VulkanRenderPath::MatrixGet(rp::MatrixStack) {
    return identity_matrix_.data();
}

// ---------------------------------------------------------------------------
// Texture loading - mirrors the bgfx path's stb_image-based decoder so the
// engine's texture pipeline can hand us pixel data through the legacy
// LoadTextureData API. Full GPU upload arrives with Phase 3.
// ---------------------------------------------------------------------------

namespace {

int* stb_pixels_to_argb(unsigned char* pixels, int w, int h) {
    int* px = new int[w * h];
    for (int i = 0; i < w * h; ++i) {
        unsigned char r = pixels[i * 4 + 0];
        unsigned char g = pixels[i * 4 + 1];
        unsigned char b = pixels[i * 4 + 2];
        unsigned char a = pixels[i * 4 + 3];
        px[i] = (a << 24) | (r << 16) | (g << 8) | b;
    }
    return px;
}

}  // namespace

int VulkanRenderPath::LoadTextureData(const char* filename, void* srcInfo,
                                      int** dataOut) {
    int w = 0, h = 0, channels = 0;
    unsigned char* pixels = stbi_load(filename, &w, &h, &channels, 4);
    if (!pixels) return -1;
    if (auto* info = static_cast<D3DXIMAGE_INFO*>(srcInfo)) {
        info->Width = w;
        info->Height = h;
    }
    *dataOut = stb_pixels_to_argb(pixels, w, h);
    stbi_image_free(pixels);
    return 0;
}

int VulkanRenderPath::LoadTextureData(uint8_t* data, uint32_t bytes,
                                      void* srcInfo, int** dataOut) {
    int w = 0, h = 0, channels = 0;
    unsigned char* pixels =
        stbi_load_from_memory(data, int(bytes), &w, &h, &channels, 4);
    if (!pixels) return -1;
    if (auto* info = static_cast<D3DXIMAGE_INFO*>(srcInfo)) {
        info->Width = w;
        info->Height = h;
    }
    *dataOut = stb_pixels_to_argb(pixels, w, h);
    stbi_image_free(pixels);
    return 0;
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

std::unique_ptr<rp::IRenderPath> make_vulkan_render_path(SDL_Window* window) {
    return std::make_unique<VulkanRenderPath>(window);
}
