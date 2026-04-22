#include "FrameContext.h"
#include "VkCheck.h"

#include <algorithm>
#include <cassert>
#include <cstdio>

namespace plce::vk {

void FrameContext::recreate_transient(VmaAllocator alloc, VkDeviceSize new_size) {
    transient.reset();
    transient_mapped_ = nullptr;

    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size  = new_size;
    bci.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    VmaAllocationCreateInfo vai{};
    vai.usage = VMA_MEMORY_USAGE_AUTO;
    vai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VmaAllocationInfo info{};
    VkBuffer      buf   = VK_NULL_HANDLE;
    VmaAllocation a     = nullptr;
    check(vmaCreateBuffer(alloc, &bci, &vai, &buf, &a, &info), "transient vb");
    transient = VmaBuffer(alloc, buf, a, info.pMappedData);
    transient_mapped_ = static_cast<std::byte*>(info.pMappedData);
    transient_size_ = new_size;
}

void FrameContext::create(VkDevice dev, VmaAllocator alloc, uint32_t queue_family,
                          float timestamp_period_ns) {
    // Command pool + primary command buffer. No RESET_COMMAND_BUFFER_BIT —
    // we reset the whole pool once per frame instead of the single CB,
    // which is cheaper in the driver and scales to additional CBs later
    // without revisiting pool flags. TRANSIENT_BIT hints that every
    // buffer allocated here is short-lived (one frame) so the driver can
    // pick a lighter-weight allocator.
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    pci.queueFamilyIndex = queue_family;
    check(vkCreateCommandPool(dev, &pci, nullptr, &pool), "cmd pool");

    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    check(vkAllocateCommandBuffers(dev, &ai, &cmd), "cmd buf");

    // Sync objects — fence starts signaled so first wait succeeds
    VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    check(vkCreateSemaphore(dev, &sci, nullptr, &sem_acquired), "sem acq");
    check(vkCreateSemaphore(dev, &sci, nullptr, &sem_done), "sem done");

    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    check(vkCreateFence(dev, &fci, nullptr, &fence), "fence");

    // Transient vertex buffer — host-visible, persistently mapped. Grows on demand.
    recreate_transient(alloc, kInitialTransientSize);

    // Per-frame UBO (lights, fog, gamma, global lightmap). Fixed-size block
    // mapped for CPU writes; the renderer fills it at StartFrame. Sized up
    // to 256 bytes to give headroom for small future additions without a
    // re-bind; std140 layout is 112 bytes today.
    {
        frame_ubo_size_ = 256;
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size  = frame_ubo_size_;
        bci.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        VmaAllocationCreateInfo vai{};
        vai.usage = VMA_MEMORY_USAGE_AUTO;
        vai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                    VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VmaAllocationInfo info{};
        VkBuffer      buf = VK_NULL_HANDLE;
        VmaAllocation a   = nullptr;
        check(vmaCreateBuffer(alloc, &bci, &vai, &buf, &a, &info), "frame ubo");
        frame_ubo = VmaBuffer(alloc, buf, a, info.pMappedData);
        frame_ubo_mapped_ = static_cast<std::byte*>(info.pMappedData);
    }

    // GPU timestamp query pool. Slots 0 and 1 hold the frame's begin/end
    // timestamps; slots 2..kMaxTimestamps-1 feed nested push_timestamp /
    // pop_timestamp pairs. timestamp_period_ns == 0 means disabled (e.g.
    // device reports timestampValidBits == 0 on this queue family).
    ts_device_     = dev;
    ts_period_ns_  = timestamp_period_ns;
    if (ts_period_ns_ > 0.0f) {
        VkQueryPoolCreateInfo qci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qci.queryType  = VK_QUERY_TYPE_TIMESTAMP;
        qci.queryCount = kMaxTimestamps;
        check(vkCreateQueryPool(dev, &qci, nullptr, &ts_pool_),
              "timestamp query pool");
    }
}

void FrameContext::reset_acquire_semaphore(VkDevice dev) {
    if (sem_acquired) vkDestroySemaphore(dev, sem_acquired, nullptr);
    VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    check(vkCreateSemaphore(dev, &sci, nullptr, &sem_acquired), "sem acq (reset)");
}

void FrameContext::write_frame_ubo(const void* src, size_t bytes) {
    // Silent truncation hid layout mismatches — now an assertion.
    assert(frame_ubo_mapped_ && "FrameContext::write_frame_ubo before create()");
    assert(bytes <= frame_ubo_size_ && "FrameUBO write exceeds buffer size");
    std::memcpy(frame_ubo_mapped_, src, bytes);
}

void FrameContext::destroy(VkDevice dev, VmaAllocator /*alloc*/) {
    deletions.flush();
    frame_ubo.reset();
    frame_ubo_mapped_ = nullptr;
    transient.reset();
    if (ts_pool_)     vkDestroyQueryPool(dev, ts_pool_, nullptr);
    ts_pool_  = VK_NULL_HANDLE;
    if (fence)        vkDestroyFence(dev, fence, nullptr);
    if (sem_done)     vkDestroySemaphore(dev, sem_done, nullptr);
    if (sem_acquired) vkDestroySemaphore(dev, sem_acquired, nullptr);
    if (pool)         vkDestroyCommandPool(dev, pool, nullptr);
}

void FrameContext::begin(VkDevice dev, VmaAllocator alloc) {
    vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX);

    // Surface transient-buffer overflows from the previous frame for this
    // slot. Each overflow dropped a draw — silent misrender otherwise. The
    // grow-on-next-begin path below reallocates to cover the high-water
    // mark, but logging makes the drop visible until that kicks in.
    if (transient_overflow_count_ > 0) {
        std::fprintf(stderr,
                     "[vk] transient VB overflowed %u times last frame "
                     "(dropped draws); growing to %llu bytes\n",
                     transient_overflow_count_,
                     (unsigned long long)pending_grow_bytes_);
        transient_overflow_count_ = 0;
    }

    // Read back the last-frame timestamps from this slot's query pool.
    // The fence has signalled, so the queries are guaranteed available.
    if (ts_pool_ && ts_queries_pending_ && ts_period_ns_ > 0.0f) {
        uint64_t ts[kMaxTimestamps] = {};
        VkResult r = vkGetQueryPoolResults(
            dev, ts_pool_, 0, kMaxTimestamps,
            sizeof(ts), ts, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
        if (r == VK_SUCCESS) {
            const double ns = double(ts[1] - ts[0]) * ts_period_ns_;
            last_gpu_ms_ = ns / 1'000'000.0;

            // Aggregate per-tag ms from the nested pairs. Multiple pairs
            // with the same tag pointer sum into one entry. Cleared each
            // frame before repopulating.
            last_pass_ms_.clear();
            for (const TsPair& p : ts_pairs_pending_) {
                if (p.end_slot == 0) continue;  // unclosed — dropped
                const double pair_ns =
                    double(ts[p.end_slot] - ts[p.start_slot]) * ts_period_ns_;
                const double pair_ms = pair_ns / 1'000'000.0;
                bool merged = false;
                for (auto& e : last_pass_ms_) {
                    if (e.first == p.tag) { e.second += pair_ms; merged = true; break; }
                }
                if (!merged) last_pass_ms_.emplace_back(p.tag, pair_ms);
            }
        }
        ts_queries_pending_ = false;
    }
    ts_pairs_pending_.clear();
    ts_pairs_recording_.clear();
    ts_stack_.clear();
    ts_next_slot_ = 2;

    deletions.flush();

    // Grow the transient buffer if a prior frame overflowed. Safe here because
    // the fence has signaled — no GPU work is still reading from the old buffer.
    if (pending_grow_bytes_ > 0) {
        VkDeviceSize new_size = transient_size_ * 2;
        // Cap the doubling inside the loop so a pathological request larger
        // than kMaxTransientSize cannot wrap VkDeviceSize before the clamp.
        while (new_size < pending_grow_bytes_ && new_size < kMaxTransientSize)
            new_size *= 2;
        new_size = std::min(new_size, kMaxTransientSize);
        if (new_size > transient_size_) {
            std::fprintf(stderr, "[vk] transient VB grew from %llu to %llu bytes\n",
                         (unsigned long long)transient_size_,
                         (unsigned long long)new_size);
            recreate_transient(alloc, new_size);
        }
        pending_grow_bytes_ = 0;
    }

    vkResetFences(dev, 1, &fence);
    vkResetCommandPool(dev, pool, 0);
    transient_offset_ = 0;

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);

    // Reset the query pool each frame; vkCmdResetQueryPool records into
    // the command buffer so it is guaranteed to run before the
    // subsequent vkCmdWriteTimestamp on the GPU.
    if (ts_pool_) {
        vkCmdResetQueryPool(cmd, ts_pool_, 0, kMaxTimestamps);
    }
    ts_queries_recorded_ = false;
}

void FrameContext::begin_cmd_timestamps() {
    if (!ts_pool_) return;
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, ts_pool_, 0);
    ts_queries_recorded_ = true;
}

void FrameContext::end_cmd_timestamps() {
    if (!ts_pool_ || !ts_queries_recorded_) return;
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ts_pool_, 1);
    // Move this frame's recorded pairs into the pending slot so begin()
    // can read them back two frames from now when this FrameContext's
    // fence signals. Also close any pairs left open by a mispaired
    // caller — they'll show up as 0 ms rather than a stack imbalance.
    ts_pairs_pending_ = std::move(ts_pairs_recording_);
    ts_pairs_recording_.clear();
    ts_stack_.clear();
    ts_queries_pending_ = true;
}

void FrameContext::push_timestamp(const char* tag) {
    if (!ts_pool_ || ts_period_ns_ <= 0.0f) return;
    if (ts_next_slot_ >= kMaxTimestamps) return;  // pool full; drop pair
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        ts_pool_, ts_next_slot_);
    ts_pairs_recording_.push_back({tag, ts_next_slot_, 0});
    ts_stack_.push_back(uint32_t(ts_pairs_recording_.size() - 1));
    ++ts_next_slot_;
}

void FrameContext::pop_timestamp() {
    if (!ts_pool_ || ts_period_ns_ <= 0.0f) return;
    if (ts_stack_.empty()) return;
    if (ts_next_slot_ >= kMaxTimestamps) { ts_stack_.pop_back(); return; }
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                        ts_pool_, ts_next_slot_);
    ts_pairs_recording_[ts_stack_.back()].end_slot = ts_next_slot_;
    ts_stack_.pop_back();
    ++ts_next_slot_;
}

}  // namespace plce::vk
