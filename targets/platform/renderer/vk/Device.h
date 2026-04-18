#pragma once

#include <vulkan/vulkan.h>

struct SDL_Window;
struct VmaAllocator_T;
using VmaAllocator = VmaAllocator_T*;

namespace plce::vk {

/// Owns Vulkan instance, physical device, logical device, and VMA allocator.
/// Created once at startup, destroyed at shutdown. No per-frame state.
class Device {
public:
    explicit Device(SDL_Window* window, bool enable_validation = false);
    ~Device();

    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    [[nodiscard]] VkInstance       instance()        const noexcept { return instance_; }
    [[nodiscard]] VkPhysicalDevice physical_device() const noexcept { return phys_; }
    [[nodiscard]] VkDevice         device()          const noexcept { return device_; }
    [[nodiscard]] VkSurfaceKHR     surface()         const noexcept { return surface_; }
    [[nodiscard]] uint32_t         graphics_family() const noexcept { return graphics_family_; }
    [[nodiscard]] VkQueue          graphics_queue()  const noexcept { return graphics_queue_; }
    [[nodiscard]] VmaAllocator     allocator()       const noexcept { return allocator_; }

    /// Set a debug name on a Vulkan object (no-op in release builds).
    void set_debug_name(VkObjectType type, uint64_t handle,
                        const char* name) const;

private:
    void create_instance(SDL_Window* window, bool enable_validation);
    void create_surface(SDL_Window* window);
    void pick_physical_device();
    void create_device();
    void create_allocator();

    VkInstance                instance_        = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debug_messenger_ = VK_NULL_HANDLE;
    VkSurfaceKHR             surface_         = VK_NULL_HANDLE;
    VkPhysicalDevice         phys_            = VK_NULL_HANDLE;
    VkDevice                 device_          = VK_NULL_HANDLE;
    uint32_t                 graphics_family_ = 0;
    VkQueue                  graphics_queue_  = VK_NULL_HANDLE;
    VmaAllocator             allocator_       = nullptr;
};

}  // namespace plce::vk
