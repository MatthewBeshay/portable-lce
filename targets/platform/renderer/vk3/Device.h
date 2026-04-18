#pragma once

#include <vulkan/vulkan_raii.hpp>
#include <vma/vk_mem_alloc.h>

#include <cstdint>

struct SDL_Window;

namespace plce::vk3 {

/// Vulkan device context — owns instance, surface, device, VMA, and queue.
/// Created once at startup, destroyed at shutdown.
class Device {
public:
    struct Config {
        SDL_Window* window            = nullptr;
        bool        enable_validation = false;
    };

    explicit Device(const Config& cfg);
    ~Device();

    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    // --- Accessors (raw handles for interop with VMA / legacy code) ---
    VkInstance       instance()      const { return *instance_; }
    VkPhysicalDevice physical()     const { return *physical_; }
    VkDevice         handle()       const { return *device_; }
    VkSurfaceKHR     surface()      const { return *surface_; }
    uint32_t         queue_family() const { return queue_family_; }
    VkQueue          queue()        const { return queue_; }
    VmaAllocator     allocator()    const { return allocator_; }

    // --- RAII accessors ---
    const vk::raii::Instance&       vk_instance() const { return instance_; }
    const vk::raii::PhysicalDevice& vk_physical() const { return physical_; }
    const vk::raii::Device&         vk_device()   const { return device_; }

    /// Debug name helper (no-op in release builds).
    void name(VkObjectType type, uint64_t obj, const char* label) const;

    template <typename T>
    void name(T handle, const char* label) const {
        name(T::objectType, reinterpret_cast<uint64_t>(static_cast<typename T::CType>(handle)), label);
    }

private:
    vk::raii::Context              ctx_;
    vk::raii::Instance             instance_{nullptr};
    vk::raii::DebugUtilsMessengerEXT messenger_{nullptr};
    vk::raii::SurfaceKHR           surface_{nullptr};
    vk::raii::PhysicalDevice       physical_{nullptr};
    vk::raii::Device               device_{nullptr};
    uint32_t                       queue_family_ = 0;
    VkQueue                        queue_        = VK_NULL_HANDLE;
    VmaAllocator                   allocator_    = nullptr;
};

}  // namespace plce::vk3
