#pragma once

#include <vulkan/vulkan.h>
#include <vma/vk_mem_alloc.h>

#include <cstdint>
#include <vector>

#include "VmaResources.h"

namespace plce::vk {

class Device;

/// Swapchain + depth attachment. Handles creation, resize, and cleanup.
class Swapchain {
public:
    /// Create the swapchain. `preferred_mode` is the desired present mode —
    /// FIFO is always available and used as fallback if unsupported.
    /// `old_swapchain`, if non-null, is passed as
    /// VkSwapchainCreateInfoKHR::oldSwapchain so the driver can reuse
    /// image memory / internal state across a resize. The caller retains
    /// ownership of the old handle and must destroy it after `create`
    /// returns (resize() does this).
    void create(const Device& dev, uint32_t w, uint32_t h,
                VkPresentModeKHR preferred_mode = VK_PRESENT_MODE_FIFO_KHR,
                VkSwapchainKHR old_swapchain = VK_NULL_HANDLE);
    void destroy(const Device& dev);
    void resize(const Device& dev, uint32_t w, uint32_t h);

    VkSwapchainKHR handle()       const { return swapchain_; }
    VkFormat       format()       const { return format_; }
    VkExtent2D     extent()       const { return extent_; }
    VkFormat       depth_format() const { return kDepthFormat; }
    VkImage        depth_image()  const { return depth_image_.handle(); }
    VkImageView    depth_view()   const { return depth_view_; }

    VkImage     image(uint32_t i) const { return images_[i]; }
    VkImageView view(uint32_t i)  const { return views_[i]; }
    uint32_t    image_count()     const { return uint32_t(images_.size()); }

private:
    void create_depth(const Device& dev);
    void destroy_depth(const Device& dev);

    VkSwapchainKHR           swapchain_   = VK_NULL_HANDLE;
    VkFormat                 format_      = VK_FORMAT_UNDEFINED;
    VkExtent2D               extent_      = {};
    VkPresentModeKHR         present_mode_ = VK_PRESENT_MODE_FIFO_KHR;
    std::vector<VkImage>     images_;
    std::vector<VkImageView> views_;

    static constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;
    VmaImage    depth_image_;
    VkImageView depth_view_ = VK_NULL_HANDLE;
};

}  // namespace plce::vk
