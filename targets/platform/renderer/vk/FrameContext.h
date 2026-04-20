#pragma once

#include <vulkan/vulkan.h>
#include <vma/vk_mem_alloc.h>

#include <cstdint>
#include <cstring>

#include "DeletionQueue.h"

namespace plce::vk {

/// Per-frame GPU resources. The renderer owns kFramesInFlight of these
/// and rotates between them.
class FrameContext {
public:
    static constexpr VkDeviceSize kInitialTransientSize = 64ull * 1024 * 1024;  // 64 MB
    static constexpr VkDeviceSize kMaxTransientSize     = 512ull * 1024 * 1024; // 512 MB safety ceiling

    void create(VkDevice dev, VmaAllocator alloc, uint32_t queue_family);
    void destroy(VkDevice dev, VmaAllocator alloc);

    /// Wait for this frame's fence, flush deletions, reset for new recording.
    /// Grows the transient buffer if a previous frame recorded an overflow.
    void begin(VkDevice dev, VmaAllocator alloc);

    /// End command buffer recording.
    void end_cmd() { vkEndCommandBuffer(cmd); }

    /// Bump-allocate from the transient vertex buffer.
    /// Returns nullptr if the allocation doesn't fit. When this happens, a
    /// grow request is recorded so the next begin() for this slot reallocates
    /// the buffer with enough headroom. The current draw is still dropped.
    void* alloc_transient(VkDeviceSize bytes) {
        if (transient_offset_ + bytes > transient_size_) {
            // Track the high-water mark so begin() knows how big to grow to.
            VkDeviceSize needed = transient_offset_ + bytes;
            if (needed > pending_grow_bytes_) pending_grow_bytes_ = needed;
            return nullptr;
        }
        void* p = transient_mapped_ + transient_offset_;
        transient_offset_ += bytes;
        return p;
    }

    VkDeviceSize transient_pos()  const { return transient_offset_; }
    VkDeviceSize transient_size() const { return transient_size_; }

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
    void recreate_transient(VmaAllocator alloc, VkDeviceSize new_size);

    std::byte*   transient_mapped_   = nullptr;
    VkDeviceSize transient_offset_   = 0;
    VkDeviceSize transient_size_     = 0;
    VkDeviceSize pending_grow_bytes_ = 0;  // >0 means begin() will reallocate
};

}  // namespace plce::vk
