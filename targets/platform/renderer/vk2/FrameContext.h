#pragma once

#include <vulkan/vulkan.h>
#include <vma/vk_mem_alloc.h>

#include <cstdint>
#include <cstring>

#include "DeletionQueue.h"

namespace plce::vk2 {

/// Per-frame GPU resources. The renderer owns kFramesInFlight of these
/// and rotates between them. Each frame context owns its own:
///   - command pool + primary command buffer
///   - fence (signaled when GPU finishes this frame)
///   - semaphores (acquire + render-done)
///   - transient vertex buffer (bump-allocated per frame)
///   - deletion queue (flushed when fence signals)
class FrameContext {
public:
    static constexpr VkDeviceSize kTransientSize = 64ull * 1024 * 1024;

    void create(VkDevice dev, VmaAllocator alloc, uint32_t queue_family);
    void destroy(VkDevice dev, VmaAllocator alloc);

    /// Wait for this frame's fence (blocks until GPU is done), then
    /// flush the deletion queue and reset transient state.
    void begin(VkDevice dev);

    /// End command buffer recording.
    void end_cmd() { vkEndCommandBuffer(cmd); }

    /// Allocate bytes from the transient vertex buffer. Returns nullptr
    /// if the buffer is full (draw should be skipped).
    void* alloc_transient(VkDeviceSize bytes) {
        if (transient_offset + bytes > kTransientSize) return nullptr;
        void* p = transient_mapped + transient_offset;
        transient_offset += bytes;
        return p;
    }

    VkDeviceSize transient_pos() const { return transient_offset; }

    // Public members for direct access (thin wrapper, no getters needed)
    VkCommandPool   pool   = VK_NULL_HANDLE;
    VkCommandBuffer cmd    = VK_NULL_HANDLE;
    VkSemaphore     sem_acquired = VK_NULL_HANDLE;
    VkSemaphore     sem_done     = VK_NULL_HANDLE;
    VkFence         fence        = VK_NULL_HANDLE;
    VkBuffer        transient_vb = VK_NULL_HANDLE;
    VmaAllocation   transient_alloc = nullptr;
    DeletionQueue   deletions;

private:
    std::byte*   transient_mapped  = nullptr;
    VkDeviceSize transient_offset  = 0;
};

}  // namespace plce::vk2
