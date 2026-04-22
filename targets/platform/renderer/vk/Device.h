#pragma once

#include <vulkan/vulkan_raii.hpp>
#include <vma/vk_mem_alloc.h>

#include <atomic>
#include <cstdint>
#include <mutex>

struct SDL_Window;

namespace plce::vk {

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
    const ::vk::raii::Instance&       vk_instance() const { return instance_; }
    const ::vk::raii::PhysicalDevice& vk_physical() const { return physical_; }
    const ::vk::raii::Device&         vk_device()   const { return device_; }

    /// Debug name helper (no-op in release builds).
    void name(VkObjectType type, uint64_t obj, const char* label) const;

    template <typename T>
    void name(T handle, const char* label) const {
        name(T::objectType, reinterpret_cast<uint64_t>(static_cast<typename T::CType>(handle)), label);
    }

    /// Externally synchronized wrapper around vkQueueSubmit2. All renderer
    /// code that submits to the graphics queue goes through this so that
    /// main-thread presentation and worker-thread texture uploads cannot
    /// race on the VkQueue handle (VUID-vkQueueSubmit2-queue-parameter).
    VkResult submit2(uint32_t submit_count, const VkSubmitInfo2* submits,
                     VkFence fence) const;

    /// Externally synchronized wrapper around vkQueuePresentKHR, paired with
    /// submit2 above so present cannot race with a worker-thread upload.
    VkResult present_khr(const VkPresentInfoKHR* present_info) const;

    /// True when the physical device supports VkPhysicalDeviceFeatures::wideLines.
    /// Renderer uses this to clamp requested line widths to 1.0f on devices
    /// that reject values > 1.0.
    bool wide_lines_enabled() const { return wide_lines_enabled_; }

    /// Nanoseconds per tick of VkQueryType::TIMESTAMP on the graphics
    /// queue, or 0.0 if timestamps are unsupported on this queue family.
    /// Callers multiply by (end - begin) to get elapsed nanoseconds.
    float timestamp_period_ns() const { return timestamp_period_ns_; }

    /// Shared upload timeline semaphore. TextureManager signals a
    /// monotonically-increasing value per upload; consumers poll
    /// vkGetSemaphoreCounterValue(device, handle(), ...) once and
    /// compare against the per-upload stored value. Replaces the
    /// per-upload VkFence + fence_pool that preceded it.
    VkSemaphore upload_timeline() const { return upload_timeline_; }

    /// Atomically reserve the next timeline value for an upload. The
    /// returned value is the one to pass as the signal value in the
    /// upload's submit info.
    uint64_t next_upload_timeline_value() const {
        return ++upload_timeline_counter_;
    }

    /// Shared frame-submission timeline semaphore. Renderer signals a
    /// monotonically-increasing value at the end of each frame's graphics
    /// submit; the next StartFrame through that FrameContext slot waits
    /// on the previously-signalled value. Replaces the per-slot VkFence.
    VkSemaphore frame_timeline() const { return frame_timeline_; }
    uint64_t next_frame_timeline_value() const {
        return ++frame_timeline_counter_;
    }

private:
    ::vk::raii::Context              ctx_;
    ::vk::raii::Instance             instance_{nullptr};
    ::vk::raii::DebugUtilsMessengerEXT messenger_{nullptr};
    ::vk::raii::SurfaceKHR           surface_{nullptr};
    ::vk::raii::PhysicalDevice       physical_{nullptr};
    ::vk::raii::Device               device_{nullptr};
    uint32_t                       queue_family_ = 0;
    VkQueue                        queue_        = VK_NULL_HANDLE;
    VmaAllocator                   allocator_    = nullptr;
    mutable std::mutex             queue_mutex_;  // guards submit2
    bool                           wide_lines_enabled_ = false;
    float                          timestamp_period_ns_ = 0.0f;
    ::vk::raii::Semaphore          upload_timeline_raii_{nullptr};
    VkSemaphore                    upload_timeline_     = VK_NULL_HANDLE;
    mutable std::atomic<uint64_t>  upload_timeline_counter_{0};

    ::vk::raii::Semaphore          frame_timeline_raii_{nullptr};
    VkSemaphore                    frame_timeline_      = VK_NULL_HANDLE;
    mutable std::atomic<uint64_t>  frame_timeline_counter_{0};
};

}  // namespace plce::vk
