#include "Swapchain.h"
#include "Device.h"
#include "VkCheck.h"

#include <algorithm>
#include <cstdio>

namespace plce::vk {

void Swapchain::create(const Device& dev, uint32_t w, uint32_t h,
                       VkPresentModeKHR preferred_mode,
                       VkSwapchainKHR old_swapchain) {
    present_mode_ = preferred_mode;

    VkSurfaceCapabilitiesKHR caps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(dev.physical(), dev.surface(), &caps);

    // Pick B8G8R8A8_UNORM with SRGB_NONLINEAR color space (most common desktop)
    uint32_t fmt_n = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(dev.physical(), dev.surface(), &fmt_n, nullptr);
    std::vector<VkSurfaceFormatKHR> fmts(fmt_n);
    vkGetPhysicalDeviceSurfaceFormatsKHR(dev.physical(), dev.surface(), &fmt_n, fmts.data());

    VkSurfaceFormatKHR chosen = fmts.front();
    for (auto& f : fmts) {
        if (f.format == VK_FORMAT_B8G8R8A8_UNORM &&
            f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            chosen = f;
            break;
        }
    }
    format_ = chosen.format;

    // Determine extent
    VkExtent2D ext{w, h};
    if (caps.currentExtent.width != UINT32_MAX)
        ext = caps.currentExtent;
    ext.width  = std::clamp(ext.width,  caps.minImageExtent.width,  caps.maxImageExtent.width);
    ext.height = std::clamp(ext.height, caps.minImageExtent.height, caps.maxImageExtent.height);
    extent_ = ext;

    uint32_t img_count = caps.minImageCount + 1;
    if (caps.maxImageCount > 0)
        img_count = std::min(img_count, caps.maxImageCount);

    VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    ci.surface          = dev.surface();
    ci.minImageCount    = img_count;
    ci.imageFormat      = format_;
    ci.imageColorSpace  = chosen.colorSpace;
    ci.imageExtent      = extent_;
    ci.imageArrayLayers = 1;
    ci.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.preTransform     = caps.currentTransform;
    ci.compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;

    // Select present mode: use requested mode if supported, else fall back to FIFO
    // (which is always guaranteed available per Vulkan spec).
    VkPresentModeKHR mode = VK_PRESENT_MODE_FIFO_KHR;
    if (preferred_mode != VK_PRESENT_MODE_FIFO_KHR) {
        uint32_t pm_n = 0;
        vkGetPhysicalDeviceSurfacePresentModesKHR(dev.physical(), dev.surface(), &pm_n, nullptr);
        std::vector<VkPresentModeKHR> modes(pm_n);
        vkGetPhysicalDeviceSurfacePresentModesKHR(dev.physical(), dev.surface(), &pm_n, modes.data());
        for (auto m : modes) if (m == preferred_mode) { mode = m; break; }
    }
    ci.presentMode      = mode;
    ci.clipped          = VK_TRUE;
    ci.oldSwapchain     = old_swapchain;
    check(vkCreateSwapchainKHR(dev.handle(), &ci, nullptr, &swapchain_),
          "vkCreateSwapchainKHR");

    // Retrieve images + create views
    uint32_t n = 0;
    vkGetSwapchainImagesKHR(dev.handle(), swapchain_, &n, nullptr);
    images_.resize(n);
    vkGetSwapchainImagesKHR(dev.handle(), swapchain_, &n, images_.data());
    views_.resize(n);
    for (uint32_t i = 0; i < n; ++i) {
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image    = images_[i];
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format   = format_;
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        check(vkCreateImageView(dev.handle(), &vi, nullptr, &views_[i]),
              "swapchain view");
    }

    create_depth(dev);
}

void Swapchain::destroy(const Device& dev) {
    destroy_depth(dev);
    for (auto v : views_) vkDestroyImageView(dev.handle(), v, nullptr);
    views_.clear();
    images_.clear();
    if (swapchain_) vkDestroySwapchainKHR(dev.handle(), swapchain_, nullptr);
    swapchain_ = VK_NULL_HANDLE;
}

void Swapchain::resize(const Device& dev, uint32_t w, uint32_t h) {
    // Hand the old swapchain to vkCreateSwapchainKHR so the driver can
    // reuse image allocations and keep presenting until the new chain is
    // ready (avoids the black frame that a raw destroy+create produces).
    // Views, depth, and the old swapchain itself are destroyed *after* the
    // new one is built.
    //
    // Wait idle before touching the old views — on vendors that keep a
    // present in flight after swapchain creation (AMD, some mobile), a
    // frame in the compositor queue can still reference a view we're
    // about to destroy. The callsite may also resize mid-frame. Pay the
    // one-shot stall here; resize is not a hot path.
    vkDeviceWaitIdle(dev.handle());

    VkSwapchainKHR   old_sc   = swapchain_;
    VkPresentModeKHR mode     = present_mode_;
    destroy_depth(dev);
    for (auto v : views_) vkDestroyImageView(dev.handle(), v, nullptr);
    views_.clear();
    images_.clear();
    swapchain_ = VK_NULL_HANDLE;

    create(dev, w, h, mode, old_sc);

    if (old_sc) vkDestroySwapchainKHR(dev.handle(), old_sc, nullptr);
}

VkFormat Swapchain::pick_depth_format(const Device& dev) {
    // Prefer a format that includes a stencil aspect so stencil features
    // (StateSetStencil) work without a second reformat later. D32_SFLOAT
    // stays as a last resort — stencil paths will then be wired into the
    // first depth format that carries a stencil aspect.
    const VkFormat candidates[] = {
        VK_FORMAT_D32_SFLOAT_S8_UINT,
        VK_FORMAT_D24_UNORM_S8_UINT,
        VK_FORMAT_D32_SFLOAT,
    };
    for (VkFormat f : candidates) {
        VkFormatProperties p{};
        vkGetPhysicalDeviceFormatProperties(dev.physical(), f, &p);
        if (p.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)
            return f;
    }
    return VK_FORMAT_D32_SFLOAT;
}

void Swapchain::create_depth(const Device& dev) {
    depth_format_ = pick_depth_format(dev);

    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType   = VK_IMAGE_TYPE_2D;
    ici.format      = depth_format_;
    ici.extent      = {extent_.width, extent_.height, 1};
    ici.mipLevels   = 1;
    ici.arrayLayers = 1;
    ici.samples     = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling      = VK_IMAGE_TILING_OPTIMAL;
    ici.usage       = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO;
    VkImage       img   = VK_NULL_HANDLE;
    VmaAllocation alloc = nullptr;
    check(vmaCreateImage(dev.allocator(), &ici, &ai, &img, &alloc, nullptr),
          "depth image");
    depth_image_ = VmaImage(dev.allocator(), img, alloc);

    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
    if (has_stencil()) aspect |= VK_IMAGE_ASPECT_STENCIL_BIT;

    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image    = depth_image_.handle();
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format   = depth_format_;
    vi.subresourceRange = {aspect, 0, 1, 0, 1};
    check(vkCreateImageView(dev.handle(), &vi, nullptr, &depth_view_),
          "depth view");
}

void Swapchain::destroy_depth(const Device& dev) {
    if (depth_view_) vkDestroyImageView(dev.handle(), depth_view_, nullptr);
    depth_view_ = VK_NULL_HANDLE;
    depth_image_.reset();
}

}  // namespace plce::vk
