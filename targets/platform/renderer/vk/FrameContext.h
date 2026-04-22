#pragma once

#include <vulkan/vulkan.h>
#include <vma/vk_mem_alloc.h>

#include <cstdint>
#include <cstring>
#include <span>
#include <utility>
#include <vector>

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

    // Per-frame UBO ring dimensions. 256-byte stride trivially satisfies
    // every minUniformBufferOffsetAlignment we target (≤256 on all known
    // desktop GPUs) and leaves headroom for FrameUBO growth. 64 slots
    // covers a handful of views + nested sub-views with margin.
    static constexpr uint32_t kFrameUboSlots       = 64;
    static constexpr uint32_t kFrameUboSlotStride  = 256;

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
    /// end_cmd_timestamps() just before vkEndCommandBuffer. These reserve
    /// slots 0 and 1; push_timestamp / pop_timestamp nest inside and
    /// consume slots 2..kMaxTimestamps.
    void begin_cmd_timestamps();
    void end_cmd_timestamps();

    /// Begin a named GPU timing pair inside command recording. Pairs may
    /// nest. `tag` is stored by pointer — pass a string literal or another
    /// pointer that outlives the frame. Silently no-ops if timestamps are
    /// disabled on this device or if the pool has no free slots.
    void push_timestamp(const char* tag);
    /// Close the most recently opened pair (LIFO with push_timestamp).
    void pop_timestamp();

    /// Most recent completed frame's whole-GPU-time in milliseconds, or 0
    /// if no measurement is available yet.
    [[nodiscard]] double last_frame_gpu_ms() const { return last_gpu_ms_; }

    /// Per-tag ms accumulated over the last completed frame. Multiple
    /// push_timestamp calls with the same tag in one frame sum into one
    /// entry. Empty until the first readback completes.
    [[nodiscard]] std::span<const std::pair<const char*, double>>
    last_pass_ms() const { return last_pass_ms_; }

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
    /// the buffer with enough headroom. The current draw is still dropped —
    /// begin() logs the overflow count at the start of the next frame so
    /// dropped geometry is visible rather than silent.
    void* alloc_transient(VkDeviceSize bytes) {
        if (transient_offset_ + bytes > transient_size_) {
            // Track the high-water mark so begin() knows how big to grow to.
            VkDeviceSize needed = transient_offset_ + bytes;
            if (needed > pending_grow_bytes_) pending_grow_bytes_ = needed;
            ++transient_overflow_count_;
            return nullptr;
        }
        void* p = transient_mapped_ + transient_offset_;
        transient_offset_ += bytes;
        return p;
    }

    VkDeviceSize transient_pos()  const { return transient_offset_; }
    VkDeviceSize transient_size() const { return transient_size_; }

    /// Allocate one FrameUBO-sized slot inside this frame's ring, copy
    /// `bytes` from `src` into it, and return the byte offset of the
    /// slot (for use as a dynamic UBO offset at bind time). Returns 0
    /// and overwrites the first slot if the ring is exhausted this
    /// frame; the caller also gets a stderr warning. Slot stride is
    /// kFrameUboSlotStride (256 bytes, std140-friendly for any UBO
    /// layout up to that size).
    [[nodiscard]] uint32_t alloc_frame_ubo_slot(const void* src, size_t bytes);

    // Public handles for direct access
    VkCommandPool   pool         = VK_NULL_HANDLE;
    VkCommandBuffer cmd          = VK_NULL_HANDLE;
    VkSemaphore     sem_acquired = VK_NULL_HANDLE;
    VkSemaphore     sem_done     = VK_NULL_HANDLE;
    VkFence         fence        = VK_NULL_HANDLE;
    VmaBuffer       transient;   // host-visible, persistently mapped
    VmaBuffer       frame_ubo;   // host-visible, persistently mapped
    VkDescriptorSet frame_ubo_set = VK_NULL_HANDLE;  // owned by Renderer's pool
    DeletionQueue   deletions;

    VkBuffer transient_vb() const { return transient.handle(); }
    VkBuffer frame_ubo_buf() const { return frame_ubo.handle(); }

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
    uint32_t     transient_overflow_count_ = 0;  // dropped draws last frame

    // Per-frame UBO ring — persistently mapped. Each slot holds one
    // FrameUBO-sized block; the renderer allocates one slot per ViewDesc
    // (plus a frame-default at slot 0 filled by StartFrame). begin()
    // resets the ring cursor.
    std::byte*   frame_ubo_mapped_   = nullptr;
    VkDeviceSize frame_ubo_size_     = 0;
    uint32_t     frame_ubo_next_slot_ = 0;

    // GPU timestamp query state. One pool per FrameContext; slot 0 and 1
    // hold the frame's begin/end timestamps written by begin_cmd_timestamps
    // / end_cmd_timestamps, and the remaining slots feed nested
    // push_timestamp / pop_timestamp pairs. Queries are read back in
    // begin() on the next pass through this slot — by then the fence has
    // signalled so results are guaranteed available. timestamp_period_ns_
    // == 0 disables all timestamp recording.
    static constexpr uint32_t kMaxTimestamps = 32;

    // Pair recorded by push/pop_timestamp. `tag` is a non-owning pointer.
    struct TsPair { const char* tag; uint32_t start_slot; uint32_t end_slot; };

    VkDevice    ts_device_           = VK_NULL_HANDLE;
    VkQueryPool ts_pool_              = VK_NULL_HANDLE;
    float       ts_period_ns_         = 0.0f;
    bool        ts_queries_recorded_  = false;   // any begin_cmd_timestamps() yet?
    bool        ts_queries_pending_   = false;   // prev-pass wrote; not yet read back
    double      last_gpu_ms_          = 0.0;
    uint32_t    ts_next_slot_         = 2;       // next free slot for push_timestamp
    std::vector<TsPair>   ts_pairs_recording_;   // pairs being recorded this frame
    std::vector<TsPair>   ts_pairs_pending_;     // last frame's pairs awaiting readback
    std::vector<uint32_t> ts_stack_;             // indices into ts_pairs_recording_
    std::vector<std::pair<const char*, double>> last_pass_ms_;  // accumulated per-tag ms
};

}  // namespace plce::vk
