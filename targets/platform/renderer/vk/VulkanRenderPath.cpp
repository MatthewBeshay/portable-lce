#include "VulkanRenderPath.h"

#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>

#define VMA_IMPLEMENTATION
#include <vma/vk_mem_alloc.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <glm/gtc/matrix_transform.hpp>
#include <stdexcept>
#include <vector>

#include "platform/PlatformTypes.h"

#include "vk/shaders/basic.vert.spv.h"
#include "vk/shaders/basic.frag.spv.h"

#include "render/TerrainRenderer.h"

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

struct ThreadRecState {
    int cbuff_id = -1;
    std::vector<VulkanRenderPath::CBuffDraw> draws;
};
thread_local ThreadRecState t_rec;

VkPrimitiveTopology topology_from_legacy(int prim) {
    // Legacy GL primitive enums; treat unknowns as triangle list.
    switch (prim) {
        case 0x0000: return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;       // GL_POINTS
        case 0x0001: return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;        // GL_LINES
        case 0x0003: return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;       // GL_LINE_STRIP
        case 0x0004: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;    // GL_TRIANGLES
        case 0x0005: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;   // GL_TRIANGLE_STRIP
        case 0x0006: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;     // GL_TRIANGLE_FAN
        default:     return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// Lifecycle: construct / destruct
// ---------------------------------------------------------------------------

VulkanRenderPath::VulkanRenderPath(SDL_Window* window) : window_(window) {
    int w = 0, h = 0;
    SDL_GetWindowSize(window_, &w, &h);

#ifdef NDEBUG
    constexpr bool enable_validation = false;
#else
    constexpr bool enable_validation = true;
#endif

    create_instance(enable_validation);
    create_surface();
    pick_physical_device();
    create_device();
    create_allocator();
    create_swapchain(uint32_t(w), uint32_t(h));
    create_depth_image(swapchain_extent_.width, swapchain_extent_.height);
    create_texture_resources();
    create_pipeline_layout();
    create_quad_index_buffer();
    create_per_frame();
    default_texture_ = ensure_default_texture();
    bound_texture_   = default_texture_;

    plce::vk_render::TerrainRenderer::Config tcfg{};
    tcfg.color_format = swapchain_format_;
    tcfg.depth_format = depth_format_;
    terrain_ = std::make_unique<plce::vk_render::TerrainRenderer>(
        device_, allocator_, graphics_family_, graphics_queue_, tcfg);

    std::fprintf(stderr, "[vk] renderer=Vulkan viewport=%ux%u images=%zu\n",
                 swapchain_extent_.width, swapchain_extent_.height,
                 swapchain_views_.size());
}

VulkanRenderPath::~VulkanRenderPath() {
    if (device_) vkDeviceWaitIdle(device_);
    terrain_.reset();
    destroy_per_frame();
    destroy_quad_index_buffer();
    destroy_all_pipelines();
    destroy_pipeline_layout();
    destroy_texture_resources();
    destroy_depth_image();
    destroy_swapchain();
    if (allocator_) vmaDestroyAllocator(allocator_);
    if (device_)    vkDestroyDevice(device_, nullptr);
    if (surface_)   vkDestroySurfaceKHR(instance_, surface_, nullptr);
    if (debug_messenger_) {
        auto fn = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
            instance_, "vkDestroyDebugUtilsMessengerEXT");
        if (fn) fn(instance_, debug_messenger_, nullptr);
    }
    if (instance_) vkDestroyInstance(instance_, nullptr);
}

// ---------------------------------------------------------------------------
// Instance / surface / device
// ---------------------------------------------------------------------------

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
    ci.enabledExtensionCount = uint32_t(std::size(extensions));
    ci.ppEnabledExtensionNames = extensions;

    vk_check(vkCreateDevice(phys_, &ci, nullptr, &device_), "vkCreateDevice");
    vkGetDeviceQueue(device_, graphics_family_, 0, &graphics_queue_);
}

void VulkanRenderPath::create_allocator() {
    VmaAllocatorCreateInfo ci{};
    ci.physicalDevice = phys_;
    ci.device = device_;
    ci.instance = instance_;
    ci.vulkanApiVersion = VK_API_VERSION_1_3;
    vk_check(vmaCreateAllocator(&ci, &allocator_), "vmaCreateAllocator");
}

// ---------------------------------------------------------------------------
// Swapchain
// ---------------------------------------------------------------------------

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
    ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
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

    fb_.width  = swapchain_extent_.width;
    fb_.height = swapchain_extent_.height;
    fb_.aspect = swapchain_extent_.height > 0
                     ? float(swapchain_extent_.width) /
                           float(swapchain_extent_.height)
                     : 1.0f;
    fb_.is_widescreen = fb_.aspect > 1.5f;
    fb_.is_hi_def     = swapchain_extent_.height >= 720;
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

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

static VkShaderModule make_shader_module(VkDevice device, const uint32_t* code,
                                         size_t bytes) {
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = bytes;
    ci.pCode = code;
    VkShaderModule mod = VK_NULL_HANDLE;
    vk_check(vkCreateShaderModule(device, &ci, nullptr, &mod),
             "vkCreateShaderModule");
    return mod;
}

void VulkanRenderPath::create_pipeline_layout() {
    // Push constants:
    //   vertex   [0  .. 79]: mat4 mvp + vec3 chunk_offset + 4 pad
    //   fragment [80 .. 111]: u32 textured + 12 pad + vec4 state_colour
    VkPushConstantRange pcs[2]{};
    pcs[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pcs[0].offset = 0;
    pcs[0].size = 80;
    pcs[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    pcs[1].offset = 80;
    pcs[1].size = 32;

    VkPipelineLayoutCreateInfo lci{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    lci.setLayoutCount = 1;
    lci.pSetLayouts = &tex_set_layout_;
    lci.pushConstantRangeCount = 2;
    lci.pPushConstantRanges = pcs;
    vk_check(vkCreatePipelineLayout(device_, &lci, nullptr, &pipeline_layout_),
             "vkCreatePipelineLayout");
}

void VulkanRenderPath::destroy_pipeline_layout() {
    if (pipeline_layout_) {
        vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
        pipeline_layout_ = VK_NULL_HANDLE;
    }
}

void VulkanRenderPath::destroy_all_pipelines() {
    for (auto& [k, p] : pipeline_cache_) {
        vkDestroyPipeline(device_, p, nullptr);
    }
    pipeline_cache_.clear();
}

VkPipeline VulkanRenderPath::ensure_pipeline(const PsoKey& key) {
    auto it = pipeline_cache_.find(key);
    if (it != pipeline_cache_.end()) return it->second;

    VkShaderModule vs = make_shader_module(device_, kBasicVertSpv,
                                           sizeof(kBasicVertSpv));
    VkShaderModule fs = make_shader_module(device_, kBasicFragSpv,
                                           sizeof(kBasicFragSpv));

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vs;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fs;
    stages[1].pName = "main";

    VkVertexInputBindingDescription binding{};
    binding.binding = 0;
    binding.stride = 32;
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    std::array<VkVertexInputAttributeDescription, 3> attrs{};
    attrs[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0};
    attrs[1] = {1, 0, VK_FORMAT_R32G32_SFLOAT,    12};
    // Tesselator packs colour as `(r<<24)|(g<<16)|(b<<8)|a`; the shader
    // re-shuffles the bytes back into RGBA order.
    attrs[2] = {2, 0, VK_FORMAT_R8G8B8A8_UNORM,   20};

    VkPipelineVertexInputStateCreateInfo vi{
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vi.vertexBindingDescriptionCount = 1;
    vi.pVertexBindingDescriptions = &binding;
    vi.vertexAttributeDescriptionCount = uint32_t(attrs.size());
    vi.pVertexAttributeDescriptions = attrs.data();

    VkPipelineInputAssemblyStateCreateInfo ia{
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vp{
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = 1;
    vp.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rs{
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = key.cull_back ? VK_CULL_MODE_BACK_BIT : VK_CULL_MODE_NONE;
    // We Y-flip in the shader so vertex winding stays GL-style.
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo ds{
        VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.depthTestEnable = key.depth_test ? VK_TRUE : VK_FALSE;
    ds.depthWriteEnable = key.depth_write ? VK_TRUE : VK_FALSE;
    ds.depthCompareOp = VkCompareOp(key.depth_func);

    VkPipelineColorBlendAttachmentState att{};
    att.blendEnable = key.blend_enable ? VK_TRUE : VK_FALSE;
    att.srcColorBlendFactor = VkBlendFactor(key.blend_src);
    att.dstColorBlendFactor = VkBlendFactor(key.blend_dst);
    att.colorBlendOp = VK_BLEND_OP_ADD;
    att.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    att.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    att.alphaBlendOp = VK_BLEND_OP_ADD;
    att.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                         VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo cb{
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1;
    cb.pAttachments = &att;

    VkDynamicState dyn_states[] = {VK_DYNAMIC_STATE_VIEWPORT,
                                   VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyn{
        VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dyn.dynamicStateCount = uint32_t(std::size(dyn_states));
    dyn.pDynamicStates = dyn_states;

    VkPipelineRenderingCreateInfo prci{
        VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    prci.colorAttachmentCount = 1;
    prci.pColorAttachmentFormats = &swapchain_format_;
    prci.depthAttachmentFormat   = depth_format_;

    VkGraphicsPipelineCreateInfo gci{
        VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gci.pNext = &prci;
    gci.stageCount = 2;
    gci.pStages = stages;
    gci.pVertexInputState = &vi;
    gci.pInputAssemblyState = &ia;
    gci.pViewportState = &vp;
    gci.pRasterizationState = &rs;
    gci.pMultisampleState = &ms;
    gci.pDepthStencilState = &ds;
    gci.pColorBlendState = &cb;
    gci.pDynamicState = &dyn;
    gci.layout = pipeline_layout_;
    VkPipeline pipeline = VK_NULL_HANDLE;
    vk_check(vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &gci,
                                       nullptr, &pipeline),
             "vkCreateGraphicsPipelines");

    vkDestroyShaderModule(device_, vs, nullptr);
    vkDestroyShaderModule(device_, fs, nullptr);

    pipeline_cache_.emplace(key, pipeline);
    return pipeline;
}

void VulkanRenderPath::create_depth_image(uint32_t width, uint32_t height) {
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = depth_format_;
    ici.extent = {width, height, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO;
    vk_check(vmaCreateImage(allocator_, &ici, &ai, &depth_image_,
                            &depth_alloc_, nullptr),
             "vmaCreateImage(depth)");

    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = depth_image_;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = depth_format_;
    vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    vci.subresourceRange.levelCount = 1;
    vci.subresourceRange.layerCount = 1;
    vk_check(vkCreateImageView(device_, &vci, nullptr, &depth_view_),
             "vkCreateImageView(depth)");
}

void VulkanRenderPath::destroy_depth_image() {
    if (depth_view_)  vkDestroyImageView(device_, depth_view_, nullptr);
    if (depth_image_) vmaDestroyImage(allocator_, depth_image_, depth_alloc_);
    depth_view_ = VK_NULL_HANDLE;
    depth_image_ = VK_NULL_HANDLE;
    depth_alloc_ = nullptr;
}

void VulkanRenderPath::create_quad_index_buffer() {
    // Six indices per quad: {0,1,2, 0,2,3} relative to the quad's first vertex.
    std::vector<uint32_t> indices(kMaxQuadsPerDraw * 6);
    for (uint32_t q = 0; q < kMaxQuadsPerDraw; ++q) {
        uint32_t base = q * 4;
        indices[q * 6 + 0] = base + 0;
        indices[q * 6 + 1] = base + 1;
        indices[q * 6 + 2] = base + 2;
        indices[q * 6 + 3] = base + 0;
        indices[q * 6 + 4] = base + 2;
        indices[q * 6 + 5] = base + 3;
    }
    VkDeviceSize bytes = indices.size() * sizeof(uint32_t);

    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = bytes;
    bci.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO;
    aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VmaAllocationInfo info{};
    vk_check(vmaCreateBuffer(allocator_, &bci, &aci, &quad_index_buffer_,
                             &quad_index_alloc_, &info),
             "vmaCreateBuffer(quad_index)");
    std::memcpy(info.pMappedData, indices.data(), bytes);
}

void VulkanRenderPath::destroy_quad_index_buffer() {
    if (quad_index_buffer_) {
        vmaDestroyBuffer(allocator_, quad_index_buffer_, quad_index_alloc_);
        quad_index_buffer_ = VK_NULL_HANDLE;
        quad_index_alloc_  = nullptr;
    }
}

// ---------------------------------------------------------------------------
// Textures
// ---------------------------------------------------------------------------

void VulkanRenderPath::create_texture_resources() {
    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.magFilter = VK_FILTER_NEAREST;
    si.minFilter = VK_FILTER_NEAREST;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    si.maxLod = 1.0f;
    vk_check(vkCreateSampler(device_, &si, nullptr, &tex_sampler_),
             "vkCreateSampler");

    VkDescriptorSetLayoutBinding b{};
    b.binding = 0;
    b.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b.descriptorCount = 1;
    b.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutCreateInfo lci{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    lci.bindingCount = 1;
    lci.pBindings = &b;
    vk_check(vkCreateDescriptorSetLayout(device_, &lci, nullptr,
                                         &tex_set_layout_),
             "vkCreateDescriptorSetLayout");

    VkDescriptorPoolSize ps{};
    ps.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    ps.descriptorCount = 4096;
    VkDescriptorPoolCreateInfo pci{
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.maxSets = 4096;
    pci.poolSizeCount = 1;
    pci.pPoolSizes = &ps;
    vk_check(vkCreateDescriptorPool(device_, &pci, nullptr, &tex_pool_),
             "vkCreateDescriptorPool");

    // Slot 0 is reserved for "no texture / default white".
    textures_.emplace_back();
}

void VulkanRenderPath::destroy_texture_resources() {
    for (auto& t : textures_) {
        if (t.view)  vkDestroyImageView(device_, t.view, nullptr);
        if (t.image) vmaDestroyImage(allocator_, t.image, t.alloc);
    }
    textures_.clear();
    if (tex_pool_)       vkDestroyDescriptorPool(device_, tex_pool_, nullptr);
    if (tex_set_layout_) vkDestroyDescriptorSetLayout(device_, tex_set_layout_,
                                                      nullptr);
    if (tex_sampler_)    vkDestroySampler(device_, tex_sampler_, nullptr);
    tex_pool_ = VK_NULL_HANDLE;
    tex_set_layout_ = VK_NULL_HANDLE;
    tex_sampler_ = VK_NULL_HANDLE;
}

int VulkanRenderPath::ensure_default_texture() {
    int idx = TextureCreate();
    uint32_t pixel = 0xFFFFFFFFu;  // opaque white
    upload_texture(idx, 1, 1, &pixel);
    return idx;
}

void VulkanRenderPath::upload_texture(int idx, int width, int height,
                                      const void* pixels) {
    if (idx <= 0 || size_t(idx) >= textures_.size()) return;
    if (width <= 0 || height <= 0) return;

    auto& t = textures_[idx];

    // Tear down any prior allocation if the size changed.
    if (t.ready && (t.width != uint32_t(width) ||
                    t.height != uint32_t(height))) {
        vkDeviceWaitIdle(device_);
        if (t.view)  vkDestroyImageView(device_, t.view, nullptr);
        if (t.image) vmaDestroyImage(allocator_, t.image, t.alloc);
        t = {};
        t.desc_set = textures_[idx].desc_set;  // keep the descriptor set
    }

    t.width = uint32_t(width);
    t.height = uint32_t(height);

    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_B8G8R8A8_UNORM;
    ici.extent = {t.width, t.height, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                VK_IMAGE_USAGE_SAMPLED_BIT;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO;
    vk_check(vmaCreateImage(allocator_, &ici, &ai, &t.image, &t.alloc,
                            nullptr),
             "vmaCreateImage");

    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = t.image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = VK_FORMAT_B8G8R8A8_UNORM;
    vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vci.subresourceRange.levelCount = 1;
    vci.subresourceRange.layerCount = 1;
    vk_check(vkCreateImageView(device_, &vci, nullptr, &t.view),
             "vkCreateImageView");

    if (!t.desc_set) {
        VkDescriptorSetAllocateInfo dai{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dai.descriptorPool = tex_pool_;
        dai.descriptorSetCount = 1;
        dai.pSetLayouts = &tex_set_layout_;
        vk_check(vkAllocateDescriptorSets(device_, &dai, &t.desc_set),
                 "vkAllocateDescriptorSets");
    }

    VkDescriptorImageInfo dii{};
    dii.sampler = tex_sampler_;
    dii.imageView = t.view;
    dii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet = t.desc_set;
    w.dstBinding = 0;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w.pImageInfo = &dii;
    vkUpdateDescriptorSets(device_, 1, &w, 0, nullptr);

    // Upload pixels via a host-visible staging buffer + immediate copy. The
    // game uploads textures one-shot during init; this is fine here.
    VkDeviceSize bytes = VkDeviceSize(width) * height * 4;
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = bytes;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VmaAllocationCreateInfo sai{};
    sai.usage = VMA_MEMORY_USAGE_AUTO;
    sai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VkBuffer staging = VK_NULL_HANDLE;
    VmaAllocation staging_alloc = nullptr;
    VmaAllocationInfo staging_info{};
    vk_check(vmaCreateBuffer(allocator_, &bci, &sai, &staging, &staging_alloc,
                             &staging_info),
             "vmaCreateBuffer(tex staging)");
    std::memcpy(staging_info.pMappedData, pixels, bytes);

    // One-time submit cmd buffer.
    VkCommandPoolCreateInfo pci2{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci2.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    pci2.queueFamilyIndex = graphics_family_;
    VkCommandPool one_pool = VK_NULL_HANDLE;
    vkCreateCommandPool(device_, &pci2, nullptr, &one_pool);
    VkCommandBufferAllocateInfo cai{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = one_pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(device_, &cai, &cmd);
    VkCommandBufferBeginInfo bbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bbi);

    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    b.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
    b.srcAccessMask = 0;
    b.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    b.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b.image = t.image;
    b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    b.subresourceRange.levelCount = 1;
    b.subresourceRange.layerCount = 1;
    VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dep.imageMemoryBarrierCount = 1;
    dep.pImageMemoryBarriers = &b;
    vkCmdPipelineBarrier2(cmd, &dep);

    VkBufferImageCopy region{};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = {t.width, t.height, 1};
    vkCmdCopyBufferToImage(cmd, staging, t.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    b.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    b.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    b.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    b.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    vkCmdPipelineBarrier2(cmd, &dep);

    vkEndCommandBuffer(cmd);

    VkCommandBufferSubmitInfo csi{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    csi.commandBuffer = cmd;
    VkSubmitInfo2 si2{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    si2.commandBufferInfoCount = 1;
    si2.pCommandBufferInfos = &csi;
    vkQueueSubmit2(graphics_queue_, 1, &si2, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphics_queue_);

    vmaDestroyBuffer(allocator_, staging, staging_alloc);
    vkFreeCommandBuffers(device_, one_pool, 1, &cmd);
    vkDestroyCommandPool(device_, one_pool, nullptr);

    t.ready = true;
}

int VulkanRenderPath::TextureCreate() {
    std::lock_guard lk(textures_mutex_);
    textures_.emplace_back();
    ++stat_tex_creates_;
    return int(textures_.size()) - 1;
}

void VulkanRenderPath::TextureFree(int idx) {
    std::lock_guard lk(textures_mutex_);
    if (idx <= 0 || size_t(idx) >= textures_.size()) return;
    if (idx == default_texture_) return;
    // Wait for any in-flight frames to release the descriptor before we
    // tear it down; cheap enough at the texture-free rate this game has.
    vkDeviceWaitIdle(device_);
    auto& t = textures_[idx];
    if (t.view)  vkDestroyImageView(device_, t.view, nullptr);
    if (t.image) vmaDestroyImage(allocator_, t.image, t.alloc);
    t = {};
}

void VulkanRenderPath::TextureBind(int idx) {
    ++stat_tex_binds_;
    std::lock_guard lk(textures_mutex_);
    // The legacy game uses externally-generated GL texture IDs (from
    // glGenTextures_4J), not our TextureCreate, so grow on demand.
    if (idx > 0 && size_t(idx) >= textures_.size()) {
        textures_.resize(size_t(idx) + 1);
    }
    if (idx > 0 && size_t(idx) < textures_.size()) {
        bound_texture_ = idx;
    } else {
        bound_texture_ = default_texture_;
    }
}

void VulkanRenderPath::TextureData(int width, int height, void* data,
                                   int level, int /*format*/) {
    if (level != 0 || !data) return;
    int idx;
    {
        std::lock_guard lk(textures_mutex_);
        idx = bound_texture_;
        if (idx <= 0) return;
        if (size_t(idx) >= textures_.size()) {
            textures_.resize(size_t(idx) + 1);
        }
    }
    upload_texture(idx, width, height, data);
    ++stat_tex_uploads_;
}

// ---------------------------------------------------------------------------
// Per-frame command pool + transient vertex buffer
// ---------------------------------------------------------------------------

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

        // Persistent-mapped host-visible vertex buffer; reset offset per
        // frame.
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = kTransientVbSize;
        bci.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO;
        aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                    VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VmaAllocationInfo info{};
        vk_check(vmaCreateBuffer(allocator_, &bci, &aci, &f.transient_vb,
                                 &f.transient_alloc, &info),
                 "vmaCreateBuffer(transient_vb)");
        f.transient_mapped = static_cast<std::byte*>(info.pMappedData);
    }
}

void VulkanRenderPath::destroy_per_frame() {
    for (auto& f : frames_) {
        if (f.transient_vb) {
            vmaDestroyBuffer(allocator_, f.transient_vb, f.transient_alloc);
        }
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
// Frame lifecycle (dynamic rendering)
// ---------------------------------------------------------------------------

void VulkanRenderPath::begin_render_pass(PerFrame& f) {
    // Two barriers: colour UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL, depth
    // UNDEFINED -> DEPTH_ATTACHMENT_OPTIMAL.
    VkImageMemoryBarrier2 barriers[2]{};
    barriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barriers[0].srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
    barriers[0].srcAccessMask = 0;
    barriers[0].dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    barriers[0].dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    barriers[0].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barriers[0].newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    barriers[0].image = swapchain_images_[acquired_image_];
    barriers[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barriers[0].subresourceRange.levelCount = 1;
    barriers[0].subresourceRange.layerCount = 1;

    barriers[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barriers[1].srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
    barriers[1].srcAccessMask = 0;
    barriers[1].dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                               VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
    barriers[1].dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    barriers[1].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barriers[1].newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    barriers[1].image = depth_image_;
    barriers[1].subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    barriers[1].subresourceRange.levelCount = 1;
    barriers[1].subresourceRange.layerCount = 1;

    VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dep.imageMemoryBarrierCount = 2;
    dep.pImageMemoryBarriers = barriers;
    vkCmdPipelineBarrier2(f.cmd, &dep);

    VkRenderingAttachmentInfo color{
        VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    color.imageView = swapchain_views_[acquired_image_];
    color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.clearValue.color.float32[0] = clear_color_[0];
    color.clearValue.color.float32[1] = clear_color_[1];
    color.clearValue.color.float32[2] = clear_color_[2];
    color.clearValue.color.float32[3] = clear_color_[3];

    VkRenderingAttachmentInfo depth{
        VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    depth.imageView = depth_view_;
    depth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth.clearValue.depthStencil.depth = 1.0f;

    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea.offset = {0, 0};
    ri.renderArea.extent = swapchain_extent_;
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &color;
    ri.pDepthAttachment = &depth;
    vkCmdBeginRendering(f.cmd, &ri);

    VkViewport vp{0.0f, 0.0f, float(swapchain_extent_.width),
                  float(swapchain_extent_.height), 0.0f, 1.0f};
    VkRect2D sc{{0, 0}, swapchain_extent_};
    vkCmdSetViewport(f.cmd, 0, 1, &vp);
    vkCmdSetScissor(f.cmd, 0, 1, &sc);
}

void VulkanRenderPath::ensure_render_pass(PerFrame& f) {
    if (pass_active_) return;
    begin_render_pass(f);
    pass_active_ = true;
}

void VulkanRenderPath::end_render_pass(PerFrame& f) {
    vkCmdEndRendering(f.cmd);

    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    b.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    b.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    b.dstStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
    b.dstAccessMask = 0;
    b.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    b.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    b.image = swapchain_images_[acquired_image_];
    b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    b.subresourceRange.levelCount = 1;
    b.subresourceRange.layerCount = 1;
    VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dep.imageMemoryBarrierCount = 1;
    dep.pImageMemoryBarriers = &b;
    vkCmdPipelineBarrier2(f.cmd, &dep);
}

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
    f.transient_offset = 0;
    pipeline_bound_ = false;
    pso_key_dirty_ = true;
    pass_active_ = false;

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(f.cmd, &bi);

    // Render pass is started lazily so callers can record transfer/compute
    // work (e.g. TerrainRenderer staging copies + cull dispatch) between
    // StartFrame and the first draw.

    frame_active_ = true;
}

void VulkanRenderPath::Present() {
    if (!frame_active_) return;
    PerFrame& f = frames_[frame_index_];

    // If nothing drew this frame, the render pass was never begun. Start
    // it now so the swapchain image at least gets the cleared colour and
    // ends in PRESENT_SRC_KHR layout.
    ensure_render_pass(f);

    end_render_pass(f);
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

    ++stat_frames_;
    double now = double(SDL_GetTicks64()) / 1000.0;
    if (stat_window_start_secs_ == 0.0) stat_window_start_secs_ = now;
    if (now - stat_window_start_secs_ >= 1.0) {
        std::fprintf(stderr,
                     "[vk] frames=%u draws=%u textured=%u tex_create=%u "
                     "tex_upload=%u tex_bind=%u\n",
                     stat_frames_, stat_draws_total_, stat_draws_textured_,
                     stat_tex_creates_, stat_tex_uploads_, stat_tex_binds_);
        stat_frames_ = 0;
        stat_draws_total_ = 0;
        stat_draws_textured_ = 0;
        stat_tex_creates_ = 0;
        stat_tex_uploads_ = 0;
        stat_tex_binds_ = 0;
        stat_window_start_secs_ = now;
    }
}

void VulkanRenderPath::Clear(int) {
    // Clear is folded into the render pass load op in StartFrame. Mid-frame
    // clear requests are ignored for now.
}

void VulkanRenderPath::SetClearColour(const float rgba[4]) {
    clear_color_[0] = rgba[0];
    clear_color_[1] = rgba[1];
    clear_color_[2] = rgba[2];
    clear_color_[3] = rgba[3];
}

void VulkanRenderPath::render_frame(const rp::FrameDesc&) {
    // The legacy main loop drives StartFrame and Present itself - this is
    // a no-op. Reserved for future use by the new render-graph path.
}

void VulkanRenderPath::resize(uint32_t w, uint32_t h) {
    if (w == 0 || h == 0) return;
    vkDeviceWaitIdle(device_);
    destroy_swapchain();
    destroy_depth_image();
    create_swapchain(w, h);
    create_depth_image(swapchain_extent_.width, swapchain_extent_.height);
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

// ---------------------------------------------------------------------------
// Software matrix stack
// ---------------------------------------------------------------------------

void VulkanRenderPath::MatrixMode(rp::MatrixStack stack) {
    matrix_mode_ = stack;
}

static std::vector<glm::mat4>& stack_for(VulkanRenderPath*, rp::MatrixStack);

namespace {
std::vector<glm::mat4>* current_stack(rp::MatrixStack mode,
                                     std::vector<glm::mat4>& mv,
                                     std::vector<glm::mat4>& proj,
                                     std::vector<glm::mat4>& tex) {
    switch (mode) {
        case rp::MatrixStack::modelview:  return &mv;
        case rp::MatrixStack::projection: return &proj;
        case rp::MatrixStack::texture:    return &tex;
    }
    return &mv;
}
}  // namespace

void VulkanRenderPath::MatrixSetIdentity() {
    auto* s = current_stack(matrix_mode_, modelview_stack_, projection_stack_,
                            texture_stack_);
    s->back() = glm::mat4(1.0f);
}

void VulkanRenderPath::MatrixTranslate(float x, float y, float z) {
    auto* s = current_stack(matrix_mode_, modelview_stack_, projection_stack_,
                            texture_stack_);
    s->back() = glm::translate(s->back(), glm::vec3(x, y, z));
}

void VulkanRenderPath::MatrixRotate(float angle_deg, float x, float y,
                                    float z) {
    auto* s = current_stack(matrix_mode_, modelview_stack_, projection_stack_,
                            texture_stack_);
    s->back() = glm::rotate(s->back(), glm::radians(angle_deg),
                            glm::vec3(x, y, z));
}

void VulkanRenderPath::MatrixScale(float x, float y, float z) {
    auto* s = current_stack(matrix_mode_, modelview_stack_, projection_stack_,
                            texture_stack_);
    s->back() = glm::scale(s->back(), glm::vec3(x, y, z));
}

void VulkanRenderPath::MatrixPerspective(float fovy, float aspect,
                                         float zNear, float zFar) {
    auto* s = current_stack(matrix_mode_, modelview_stack_, projection_stack_,
                            texture_stack_);
    // Y flip is applied in the vertex shader, not here, so MatrixGet stays
    // consistent with the GL convention the engine expects.
    s->back() = s->back() *
                glm::perspective(glm::radians(fovy), aspect, zNear, zFar);
}

void VulkanRenderPath::MatrixOrthogonal(float left, float right, float bottom,
                                        float top, float zNear, float zFar) {
    auto* s = current_stack(matrix_mode_, modelview_stack_, projection_stack_,
                            texture_stack_);
    s->back() = s->back() * glm::ortho(left, right, bottom, top, zNear, zFar);
}

void VulkanRenderPath::MatrixPush() {
    auto* s = current_stack(matrix_mode_, modelview_stack_, projection_stack_,
                            texture_stack_);
    s->push_back(s->back());
}

void VulkanRenderPath::MatrixPop() {
    auto* s = current_stack(matrix_mode_, modelview_stack_, projection_stack_,
                            texture_stack_);
    if (s->size() > 1) s->pop_back();
}

void VulkanRenderPath::MatrixMult(float* m) {
    auto* s = current_stack(matrix_mode_, modelview_stack_, projection_stack_,
                            texture_stack_);
    glm::mat4 mat;
    std::memcpy(&mat[0][0], m, sizeof(float) * 16);
    s->back() = s->back() * mat;
}

const float* VulkanRenderPath::MatrixGet(rp::MatrixStack stack) {
    auto* s = current_stack(stack, modelview_stack_, projection_stack_,
                            texture_stack_);
    cached_matrix_get_ = s->back();
    return &cached_matrix_get_[0][0];
}

// ---------------------------------------------------------------------------
// DrawVertices
// ---------------------------------------------------------------------------

void VulkanRenderPath::DrawVertices(int primType, int count, void* data,
                                    int vertexType, int shaderType) {
    if (count <= 0 || !data) return;

    // Recording into a CBuff list? Append to per-thread storage; CBuffEnd
    // moves it into the shared pool under lock.
    if (t_rec.cbuff_id >= 0) {
        constexpr VkDeviceSize stride = 32;
        size_t bytes = size_t(count) * stride;
        CBuffDraw d;
        d.primType = primType;
        d.vertexType = vertexType;
        d.shaderType = shaderType;
        d.verts.resize(bytes);
        std::memcpy(d.verts.data(), data, bytes);
        t_rec.draws.push_back(std::move(d));
        return;
    }

    if (!frame_active_) return;
    (void)vertexType;
    (void)shaderType;

    PerFrame& f_pass = frames_[frame_index_];
    ensure_render_pass(f_pass);

    // The pipeline is fixed at TRIANGLE_LIST. Convert legacy primitives:
    //   GL_TRIANGLES (0x0004) - already triangle list, draw as-is
    //   GL_QUADS     (0x0007) - 4 verts per quad, expand via index buffer
    //   anything else - skip for now (point/line/strip/fan need more work)
    const bool is_quads     = (primType == 0x0007);
    const bool is_triangles = (primType == 0x0004);
    if (!is_quads && !is_triangles) return;
    if (is_quads && (count % 4) != 0) return;

    PerFrame& f = frames_[frame_index_];

    constexpr VkDeviceSize stride = 32;
    VkDeviceSize bytes = VkDeviceSize(count) * stride;
    if (bytes > kTransientVbSize) return;
    if (f.transient_offset + bytes > kTransientVbSize) return;

    std::memcpy(f.transient_mapped + f.transient_offset, data, bytes);
    VkDeviceSize draw_offset = f.transient_offset;
    f.transient_offset += bytes;

    if (pso_key_dirty_ || !(current_pso_key_ == last_bound_pso_key_)) {
        VkPipeline pipeline = ensure_pipeline(current_pso_key_);
        vkCmdBindPipeline(f.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        last_bound_pso_key_ = current_pso_key_;
        pso_key_dirty_ = false;
    }

    // Snapshot the active texture's descriptor under the textures lock so
    // worker-thread Bind/Data can't move the vector underneath us.
    VkDescriptorSet ds = VK_NULL_HANDLE;
    bool textured_active = false;
    {
        std::lock_guard lk(textures_mutex_);
        int tex = (bound_texture_ > 0 &&
                   size_t(bound_texture_) < textures_.size() &&
                   textures_[bound_texture_].ready)
                      ? bound_texture_ : default_texture_;
        ds = textures_[tex].desc_set;
        textured_active = (tex != default_texture_);
    }
    if (ds) {
        vkCmdBindDescriptorSets(f.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipeline_layout_, 0, 1, &ds, 0, nullptr);
    }

    vkCmdBindVertexBuffers(f.cmd, 0, 1, &f.transient_vb, &draw_offset);

    struct VertPC {
        glm::mat4 mvp;
        float     chunk_offset[3];
        float     pad;
    } vpc{};
    vpc.mvp = projection_stack_.back() * modelview_stack_.back();
    vpc.chunk_offset[0] = chunk_offset_[0];
    vpc.chunk_offset[1] = chunk_offset_[1];
    vpc.chunk_offset[2] = chunk_offset_[2];
    vkCmdPushConstants(f.cmd, pipeline_layout_, VK_SHADER_STAGE_VERTEX_BIT, 0,
                       sizeof(vpc), &vpc);

    struct FragPC {
        uint32_t textured;
        uint32_t pad[3];
        float    state_colour[4];
    } pc{};
    pc.textured = textured_active ? 1u : 0u;
    pc.state_colour[0] = state_colour_[0];
    pc.state_colour[1] = state_colour_[1];
    pc.state_colour[2] = state_colour_[2];
    pc.state_colour[3] = state_colour_[3];
    vkCmdPushConstants(f.cmd, pipeline_layout_, VK_SHADER_STAGE_FRAGMENT_BIT,
                       80, sizeof(pc), &pc);

    if (is_quads) {
        uint32_t quad_count = uint32_t(count) / 4;
        if (quad_count > kMaxQuadsPerDraw) quad_count = kMaxQuadsPerDraw;
        vkCmdBindIndexBuffer(f.cmd, quad_index_buffer_, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(f.cmd, quad_count * 6, 1, 0, 0, 0);
    } else {
        vkCmdDraw(f.cmd, uint32_t(count), 1, 0, 0);
    }
    ++stat_draws_total_;
    if (textured_active) ++stat_draws_textured_;
}

// ---------------------------------------------------------------------------
// CBuff* (display list) record/replay
//
// Recording state is per-thread: chunk meshers run on worker threads and
// each builds its own list independently of the main thread's UI work. The
// shared cbuffs_ pool is mutex-protected so End/Call see consistent state.
// ---------------------------------------------------------------------------

int VulkanRenderPath::CBuffCreate(int n) {
    std::lock_guard lk(cbuffs_mutex_);
    int first = next_cbuff_;
    int needed = n > 0 ? n : 1;
    next_cbuff_ += needed;
    if (size_t(first + needed) > cbuffs_.size()) {
        cbuffs_.resize(size_t(first + needed));
    }
    return first;
}

void VulkanRenderPath::CBuffDeleteAll() {
    std::lock_guard lk(cbuffs_mutex_);
    cbuffs_.clear();
    next_cbuff_ = 1;
    t_rec.cbuff_id = -1;
    t_rec.draws.clear();
}

void VulkanRenderPath::CBuffStart(int index, bool /*full*/) {
    t_rec.cbuff_id = index;
    t_rec.draws.clear();
}

void VulkanRenderPath::CBuffClear(int index) {
    std::lock_guard lk(cbuffs_mutex_);
    if (index < 0 || size_t(index) >= cbuffs_.size()) return;
    cbuffs_[index].draws.clear();
    cbuffs_[index].valid = false;
}

int VulkanRenderPath::CBuffSize(int index) {
    std::lock_guard lk(cbuffs_mutex_);
    if (index < 0 || size_t(index) >= cbuffs_.size()) return 0;
    return cbuffs_[index].valid ? 1 : 0;
}

void VulkanRenderPath::CBuffEnd() {
    int id = t_rec.cbuff_id;
    t_rec.cbuff_id = -1;
    if (id < 0) return;
    std::lock_guard lk(cbuffs_mutex_);
    if (size_t(id) >= cbuffs_.size()) {
        cbuffs_.resize(size_t(id) + 1);
    }
    cbuffs_[id].draws = std::move(t_rec.draws);
    cbuffs_[id].valid = !cbuffs_[id].draws.empty();
    t_rec.draws.clear();
}

bool VulkanRenderPath::CBuffCall(int index, bool /*full*/) {
    if (index < 0 || !frame_active_) return false;
    // Snapshot the draws under the lock so workers can't move the vector
    // out from under us mid-replay.
    std::vector<CBuffDraw> snapshot;
    {
        std::lock_guard lk(cbuffs_mutex_);
        if (size_t(index) >= cbuffs_.size()) return false;
        auto& cb = cbuffs_[index];
        if (!cb.valid || cb.draws.empty()) return false;
        snapshot = cb.draws;  // copy; bgfx-style pool already does this
    }
    for (auto& d : snapshot) {
        DrawVertices(d.primType, int(d.verts.size() / 32), d.verts.data(),
                     d.vertexType, d.shaderType);
    }
    return true;
}

// ---------------------------------------------------------------------------
// Texture loading - mirrors the bgfx path's stb_image-based decoder.
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
// Render state - mutates current_pso_key_ and the shader-side colour PC.
// ---------------------------------------------------------------------------

namespace {

uint8_t blend_factor_to_vk(rp::BlendFactor f) {
    using BF = rp::BlendFactor;
    switch (f) {
        case BF::zero:                     return VK_BLEND_FACTOR_ZERO;
        case BF::one:                      return VK_BLEND_FACTOR_ONE;
        case BF::src_color:                return VK_BLEND_FACTOR_SRC_COLOR;
        case BF::one_minus_src_color:      return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
        case BF::src_alpha:                return VK_BLEND_FACTOR_SRC_ALPHA;
        case BF::one_minus_src_alpha:      return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        case BF::dst_color:                return VK_BLEND_FACTOR_DST_COLOR;
        case BF::one_minus_dst_color:      return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
        case BF::dst_alpha:                return VK_BLEND_FACTOR_DST_ALPHA;
        case BF::one_minus_dst_alpha:      return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
        case BF::constant_alpha:           return VK_BLEND_FACTOR_CONSTANT_ALPHA;
        case BF::one_minus_constant_alpha: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
    }
    return VK_BLEND_FACTOR_ONE;
}

uint8_t depth_func_to_vk(rp::DepthTest f) {
    using DT = rp::DepthTest;
    switch (f) {
        case DT::off:           return VK_COMPARE_OP_ALWAYS;
        case DT::less:          return VK_COMPARE_OP_LESS;
        case DT::less_equal:    return VK_COMPARE_OP_LESS_OR_EQUAL;
        case DT::equal:         return VK_COMPARE_OP_EQUAL;
        case DT::greater:       return VK_COMPARE_OP_GREATER;
        case DT::greater_equal: return VK_COMPARE_OP_GREATER_OR_EQUAL;
        case DT::always:        return VK_COMPARE_OP_ALWAYS;
    }
    return VK_COMPARE_OP_LESS_OR_EQUAL;
}

}  // namespace

void VulkanRenderPath::StateSetColour(float r, float g, float b, float a) {
    state_colour_ = {r, g, b, a};
}

void VulkanRenderPath::StateSetDepthMask(bool e) {
    if (current_pso_key_.depth_write != e) {
        current_pso_key_.depth_write = e;
        pso_key_dirty_ = true;
    }
}

void VulkanRenderPath::StateSetDepthTestEnable(bool e) {
    if (current_pso_key_.depth_test != e) {
        current_pso_key_.depth_test = e;
        pso_key_dirty_ = true;
    }
}

void VulkanRenderPath::StateSetDepthFunc(rp::DepthTest f) {
    uint8_t v = depth_func_to_vk(f);
    if (current_pso_key_.depth_func != v) {
        current_pso_key_.depth_func = v;
        pso_key_dirty_ = true;
    }
}

void VulkanRenderPath::StateSetBlendEnable(bool e) {
    if (current_pso_key_.blend_enable != e) {
        current_pso_key_.blend_enable = e;
        pso_key_dirty_ = true;
    }
}

void VulkanRenderPath::StateSetBlendFunc(rp::BlendFactor s, rp::BlendFactor d) {
    uint8_t vs = blend_factor_to_vk(s);
    uint8_t vd = blend_factor_to_vk(d);
    if (current_pso_key_.blend_src != vs || current_pso_key_.blend_dst != vd) {
        current_pso_key_.blend_src = vs;
        current_pso_key_.blend_dst = vd;
        pso_key_dirty_ = true;
    }
}

void VulkanRenderPath::StateSetFaceCull(bool e) {
    if (current_pso_key_.cull_back != e) {
        current_pso_key_.cull_back = e;
        pso_key_dirty_ = true;
    }
}

// ---------------------------------------------------------------------------
// GPU-driven terrain hooks (Phase 4)
// ---------------------------------------------------------------------------

void VulkanRenderPath::chunk_upload(const ChunkUpload& u) {
    if (!terrain_) return;
    plce::vk_render::TerrainRenderer::ChunkKey key{u.cx, u.cy, u.cz, u.layer};
    terrain_->upload_chunk(
        key,
        glm::vec3(u.world_origin[0], u.world_origin[1], u.world_origin[2]),
        glm::vec3(u.aabb_min[0], u.aabb_min[1], u.aabb_min[2]),
        glm::vec3(u.aabb_max[0], u.aabb_max[1], u.aabb_max[2]),
        u.vertex_data, u.vertex_count, u.vertex_stride);
}

void VulkanRenderPath::chunk_destroy(int32_t cx, int32_t cy, int32_t cz,
                                     uint8_t layer) {
    if (!terrain_) return;
    terrain_->destroy_chunk({cx, cy, cz, layer});
}

void VulkanRenderPath::chunk_upload_from_cbuff(int cbuff_id,
                                               const ChunkUpload& base) {
    if (!terrain_ || cbuff_id < 0) return;
    // Walk the recorded draws under lock, copy their vertex bytes into
    // one contiguous buffer, then call upload_chunk with that buffer.
    // The caller owns the CBuff lifetime; we don't free it here.
    std::vector<std::byte> combined;
    uint32_t total_verts = 0;
    constexpr uint32_t kStride = 32;
    {
        std::lock_guard lk(cbuffs_mutex_);
        if (size_t(cbuff_id) >= cbuffs_.size()) return;
        auto& cb = cbuffs_[cbuff_id];
        if (!cb.valid) return;
        size_t total_bytes = 0;
        for (auto& d : cb.draws) total_bytes += d.verts.size();
        combined.reserve(total_bytes);
        for (auto& d : cb.draws) {
            // Only quad batches in the standard 32-byte format are valid
            // chunk geometry for Phase 4 v1.
            if (d.primType != 0x0007) continue;
            combined.insert(combined.end(), d.verts.begin(), d.verts.end());
            total_verts += uint32_t(d.verts.size() / kStride);
        }
    }
    if (total_verts == 0) return;

    ChunkUpload up = base;
    up.vertex_data   = combined.data();
    up.vertex_count  = total_verts;
    up.vertex_stride = kStride;
    chunk_upload(up);
}

void VulkanRenderPath::set_terrain_atlas(int texture_id) {
    if (!terrain_) return;
    std::lock_guard lk(textures_mutex_);
    if (texture_id <= 0 || size_t(texture_id) >= textures_.size()) return;
    auto& t = textures_[texture_id];
    if (!t.ready || !t.view) return;
    terrain_->set_atlas(t.view, tex_sampler_);
}

void VulkanRenderPath::render_terrain(const float* mvp_4x4,
                                      const float* frustum_24) {
    if (!terrain_ || !frame_active_ || !mvp_4x4 || !frustum_24) return;

    // Auto-bind the currently-bound texture as the atlas. The game does
    // a TextureBind(terrain_atlas) right before chunk rendering, so this
    // saves us a separate set_terrain_atlas() call from the game code.
    {
        std::lock_guard lk(textures_mutex_);
        int tex = bound_texture_;
        if (tex > 0 && size_t(tex) < textures_.size() && textures_[tex].ready) {
            terrain_->set_atlas(textures_[tex].view, tex_sampler_);
        } else if (default_texture_ > 0 &&
                   size_t(default_texture_) < textures_.size()) {
            terrain_->set_atlas(textures_[default_texture_].view, tex_sampler_);
        }
    }

    PerFrame& f = frames_[frame_index_];
    glm::mat4 mvp;
    std::memcpy(&mvp[0][0], mvp_4x4, sizeof(glm::mat4));
    std::array<glm::vec4, 6> frustum;
    for (int i = 0; i < 6; ++i) {
        frustum[i] = glm::vec4(frustum_24[i * 4 + 0],
                               frustum_24[i * 4 + 1],
                               frustum_24[i * 4 + 2],
                               frustum_24[i * 4 + 3]);
    }
    // Indirect draw must be inside the render pass. Copies + cull happen
    // before begin; ensure_render_pass() begins it just in time.
    ensure_render_pass(f);
    terrain_->render(f.cmd, mvp, frustum);
    // The terrain pipeline trashes the bound state; force a rebind on the
    // next legacy DrawVertices.
    pipeline_bound_ = false;
    pso_key_dirty_ = true;
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

std::unique_ptr<rp::IRenderPath> make_vulkan_render_path(SDL_Window* window) {
    return std::make_unique<VulkanRenderPath>(window);
}
