#pragma once

#include <vulkan/vulkan.h>
#include <vma/vk_mem_alloc.h>

#include <cstdint>
#include <cstring>

#include "DeletionQueue.h"

namespace plce::vk3 {

/// Per-frame GPU resources. The renderer owns kFramesInFlight of these
/// and rotates between them.
class FrameContext {
public:
    static constexpr VkDeviceSize kTransientSize = 64ull * 1024 * 1024;  // 64 MB

    void create(VkDevice dev, VmaAllocator alloc, uint32_t queue_family);
    void destroy(VkDevice dev, VmaAllocator alloc);

    /// Wait for this frame's fence, flush deletions, reset for new recording.
    void begin(VkDevice dev);

    /// End command buffer recording.
    void end_cmd() { vkEndCommandBuffer(cmd); }

    /// Bump-allocate from the transient vertex buffer.
    /// Returns nullptr if full (draw should be skipped).
    void* alloc_transient(VkDeviceSize bytes) {
        if (transient_offset_ + bytes > kTransientSize) return nullptr;
        void* p = transient_mapped_ + transient_offset_;
        transient_offset_ += bytes;
        return p;
    }

    VkDeviceSize transient_pos() const { return transient_offset_; }

    // Public handles for direct access
    VkCommandPool   pool         = VK_NULL_HANDLE;
    VkCommandBuffer cmd          = VK_NULL_HANDLE;
    VkSemaphore     sem_acquired = VK_NULL_HANDLE;
    VkSemaphore     sem_done     = VK_NULL_HANDLE;
    VkFence         fence        = VK_NULL_HANDLE;
    VkBuffer        transient_vb = VK_NULL_HANDLE;
    VmaAllocation   transient_alloc = nullptr;
    DeletionQueue   deletions;

    FrameContext() = default;
    FrameContext(const FrameContext&) = delete;
    FrameContext& operator=(const FrameContext&) = delete;
    FrameContext(FrameContext&&) noexcept = default;
    FrameContext& operator=(FrameContext&&) noexcept = default;

private:
    std::byte*   transient_mapped_  = nullptr;
    VkDeviceSize transient_offset_  = 0;
};

}  // namespace plce::vk3
