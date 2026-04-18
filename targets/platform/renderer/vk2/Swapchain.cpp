#include "Swapchain.h"
#include "Device.h"

#include <algorithm>
#include <stdexcept>

namespace plce::vk2 {

namespace {
void check(VkResult r, const char* w) {
    if (r != VK_SUCCESS) {
        char b[128]; std::snprintf(b, sizeof b, "%s: %d", w, int(r));
        throw std::runtime_error(b);
    }
}
}  // namespace

void Swapchain::create(const Device& dev, uint32_t w, uint32_t h) {
    VkSurfaceCapabilitiesKHR caps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(dev.physical(), dev.surface(),
                                              &caps);

    uint32_t fmt_n = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(dev.physical(), dev.surface(),
                                         &fmt_n, nullptr);
    std::vector<VkSurfaceFormatKHR> fmts(fmt_n);
    vkGetPhysicalDeviceSurfaceFormatsKHR(dev.physical(), dev.surface(),
                                         &fmt_n, fmts.data());

    VkSurfaceFormatKHR chosen = fmts.front();
    for (auto& f : fmts) {
        if (f.format == VK_FORMAT_B8G8R8A8_UNORM &&
            f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            chosen = f;
            break;
        }
    }
    format_ = chosen.format;

    VkExtent2D ext{w, h};
    if (caps.currentExtent.width != UINT32_MAX) ext = caps.currentExtent;
    ext.width  = std::clamp(ext.width,  caps.minImageExtent.width,
                            caps.maxImageExtent.width);
    ext.height = std::clamp(ext.height, caps.minImageExtent.height,
                            caps.maxImageExtent.height);
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
    ci.presentMode      = VK_PRESENT_MODE_FIFO_KHR;
    ci.clipped          = VK_TRUE;
    check(vkCreateSwapchainKHR(dev.handle(), &ci, nullptr, &swapchain_),
          "swapchain");
    dev.name(swapchain_, "swapchain");

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
        vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vi.subresourceRange.levelCount = 1;
        vi.subresourceRange.layerCount = 1;
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
    // No vkDeviceWaitIdle — caller must ensure GPU is idle for this
    // frame slot via per-frame fence.
    destroy(dev);
    create(dev, w, h);
}

void Swapchain::create_depth(const Device& dev) {
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType     = VK_IMAGE_TYPE_2D;
    ici.format        = depth_format_;
    ici.extent        = {extent_.width, extent_.height, 1};
    ici.mipLevels     = 1;
    ici.arrayLayers   = 1;
    ici.samples       = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
    ici.usage         = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

    VmaAllocationCreateInfo ai{};
    ai.usage = VMA_MEMORY_USAGE_AUTO;
    check(vmaCreateImage(dev.allocator(), &ici, &ai, &depth_image_,
                         &depth_alloc_, nullptr),
          "depth image");
    dev.name(depth_image_, "depth");

    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image    = depth_image_;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format   = depth_format_;
    vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    vi.subresourceRange.levelCount = 1;
    vi.subresourceRange.layerCount = 1;
    check(vkCreateImageView(dev.handle(), &vi, nullptr, &depth_view_),
          "depth view");
    dev.name(depth_view_, "depth_view");
}

void Swapchain::destroy_depth(const Device& dev) {
    if (depth_view_)  vkDestroyImageView(dev.handle(), depth_view_, nullptr);
    if (depth_image_) vmaDestroyImage(dev.allocator(), depth_image_, depth_alloc_);
    depth_view_  = VK_NULL_HANDLE;
    depth_image_ = VK_NULL_HANDLE;
    depth_alloc_ = nullptr;
}

}  // namespace plce::vk2
