#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>

struct SDL_Window;
struct VmaAllocator_T;
using VmaAllocator = VmaAllocator_T*;

namespace plce::vk2 {

/// Vulkan device context — created once at startup.
/// Owns instance, surface, physical/logical device, VMA allocator, queue.
class Device {
public:
    struct Config {
        SDL_Window* window          = nullptr;
        bool        enable_validation = false;
    };

    explicit Device(const Config& cfg);
    ~Device();

    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    // Accessors
    VkInstance       instance()        const { return instance_; }
    VkPhysicalDevice physical()       const { return phys_; }
    VkDevice         handle()         const { return device_; }
    VkSurfaceKHR     surface()        const { return surface_; }
    uint32_t         queue_family()   const { return queue_family_; }
    VkQueue          queue()          const { return queue_; }
    VmaAllocator     allocator()      const { return allocator_; }

    // Debug naming (no-op if debug utils unavailable)
    void name(VkObjectType type, uint64_t handle, const char* name) const;

    // Convenience template for naming typed handles
    template <typename T>
    void name(T h, const char* n) const {
        name(object_type<T>(), reinterpret_cast<uint64_t>(h), n);
    }

private:
    template <typename T> static constexpr VkObjectType object_type();

    VkInstance                instance_  = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT messenger_ = VK_NULL_HANDLE;
    VkSurfaceKHR             surface_   = VK_NULL_HANDLE;
    VkPhysicalDevice         phys_      = VK_NULL_HANDLE;
    VkDevice                 device_    = VK_NULL_HANDLE;
    uint32_t                 queue_family_ = 0;
    VkQueue                  queue_     = VK_NULL_HANDLE;
    VmaAllocator             allocator_ = nullptr;

    PFN_vkSetDebugUtilsObjectNameEXT pfn_set_name_ = nullptr;
};

// Object type specializations for name()
template <> constexpr VkObjectType Device::object_type<VkImage>()          { return VK_OBJECT_TYPE_IMAGE; }
template <> constexpr VkObjectType Device::object_type<VkBuffer>()         { return VK_OBJECT_TYPE_BUFFER; }
template <> constexpr VkObjectType Device::object_type<VkImageView>()      { return VK_OBJECT_TYPE_IMAGE_VIEW; }
template <> constexpr VkObjectType Device::object_type<VkPipeline>()       { return VK_OBJECT_TYPE_PIPELINE; }
template <> constexpr VkObjectType Device::object_type<VkCommandBuffer>()  { return VK_OBJECT_TYPE_COMMAND_BUFFER; }
template <> constexpr VkObjectType Device::object_type<VkDescriptorSet>()  { return VK_OBJECT_TYPE_DESCRIPTOR_SET; }
template <> constexpr VkObjectType Device::object_type<VkSampler>()        { return VK_OBJECT_TYPE_SAMPLER; }
template <> constexpr VkObjectType Device::object_type<VkSwapchainKHR>()   { return VK_OBJECT_TYPE_SWAPCHAIN_KHR; }

}  // namespace plce::vk2
