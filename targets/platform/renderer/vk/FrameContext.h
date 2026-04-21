#pragma once

#include <vulkan/vulkan.h>
#include <vma/vk_mem_alloc.h>

#include <cstdint>
#include <cstring>

#include "DeletionQueue.h"
#include "VmaResources.h"

namespace plce::vk {

/// Per-frame GPU resources. The renderer owns kFramesInFlight of these
/// and rotates between them.
class FrameContext {
public:
    // Transient vertex buffer sizing. UI / entity / debug draws are the only
    // users of the transient path (chunk meshes go through DisplayListManager),
    // so the initial size starts small and grows geometrically on overflow.
    // Upper bound is the point at which a single frame's transient draws have
    // exceeded anything sensible and something upstream is leaking.
    static constexpr VkDeviceSize kInitialTransientSize = 8ull  * 1024 * 1024;  // 8 MB
    static constexpr VkDeviceSize kMaxTransientSize     = 64ull * 1024 * 1024;  // 64 MB ceiling

    void create(VkDevice dev, VmaAllocator alloc, uint32_t queue_family,
                float timestamp_period_ns = 0.0f);
    void destroy(VkDevice dev, VmaAllocator alloc);

    /// Wait for this frame's fence, flush deletions, reset for new recording.
    /// Grows the transient buffer if a previous frame recorded an overflow.
    /// If GPU timestamp queries are enabled, reads back the last frame's
    /// timestamps and accumulates them; caller can poll via frame_gpu_ms().
    void begin(VkDevice dev, VmaAllocator alloc);

    /// Record a begin/end timestamp pair into the frame's query pool.
    /// begin_cmd_timestamps() is called at the top of command recording,
    /// end_cmd_timestamps() just before vkEndCommandBuffer.
    void begin_cmd_timestamps();
    void end_cmd_timestamps();

    /// Most recent completed frame's whole-GPU-time in milliseconds, or 0
    /// if no measurement is available yet.
    [[nodiscard]] double last_frame_gpu_ms() const { return last_gpu_ms_; }

    /// End command buffer recording.
    void end_cmd() { vkEndCommandBuffer(cmd); }

    /// Flush the range of the transient VB that was written this frame.
    /// No-op on coherent memory; required before the GPU reads the data
    /// on non-coherent iGPU / mobile paths. Call once per frame from
    /// Renderer::Present before submit2.
    void flush_transient_writes(VmaAllocator alloc) {
        if (transient_offset_ == 0 || !transient) return;
        vmaFlushAllocation(alloc, transient.allocation(), 0, transient_offset_);
    }

    /// Destroy and recreate sem_acquired. Used on swapchain recreation:
    /// after a VK_SUBOPTIMAL_KHR return, vkAcquireNextImageKHR has already
    /// signaled the semaphore, and calling acquire a second time with the
    /// same (still-signaled) semaphore is undefined behaviour. Caller must
    /// have waited on this frame's fence first.
    void reset_acquire_semaphore(VkDevice dev);

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
    VmaBuffer       transient;   // host-visible, persistently mapped
    DeletionQueue   deletions;

    VkBuffer transient_vb() const { return transient.handle(); }

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

    // GPU timestamp query state. One pool per FrameContext (2 queries:
    // begin + end of the frame's command buffer). A query is read back
    // in begin() of the next iteration through this slot — by then the
    // fence has signalled, so the query is guaranteed to be available.
    // timestamp_period_ns_ == 0 disables the feature.
    VkDevice    ts_device_           = VK_NULL_HANDLE;
    VkQueryPool ts_pool_              = VK_NULL_HANDLE;
    float       ts_period_ns_         = 0.0f;
    bool        ts_queries_recorded_  = false;   // any begin_cmd_timestamps() yet?
    bool        ts_queries_pending_   = false;   // prev-pass wrote; not yet read back
    double      last_gpu_ms_          = 0.0;
};

}  // namespace plce::vk
