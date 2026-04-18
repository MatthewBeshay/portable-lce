#pragma once

#include <vulkan/vulkan.h>
#include <vma/vk_mem_alloc.h>

#include <functional>
#include <mutex>
#include <vector>

namespace plce::vk2 {

/// Accumulates GPU resources for deferred destruction.
/// Resources are destroyed only after their frame's fence has signaled,
/// guaranteeing the GPU is no longer using them. This eliminates every
/// vkDeviceWaitIdle call on hot paths.
class DeletionQueue {
public:
    using Fn = std::function<void()>;

    /// Push a cleanup function to run when flush() is called.
    void push(Fn&& fn) {
        std::lock_guard lk(mu_);
        pending_.push_back(std::move(fn));
    }

    /// Convenience: schedule a VkBuffer + VmaAllocation for destruction.
    void push_buffer(VmaAllocator alloc, VkBuffer buf, VmaAllocation a) {
        if (!buf) return;
        push([=]() { vmaDestroyBuffer(alloc, buf, a); });
    }

    /// Convenience: schedule a VkImage + VmaAllocation for destruction.
    void push_image(VmaAllocator alloc, VkImage img, VmaAllocation a) {
        if (!img) return;
        push([=]() { vmaDestroyImage(alloc, img, a); });
    }

    /// Convenience: schedule a VkImageView for destruction.
    void push_view(VkDevice dev, VkImageView v) {
        if (!v) return;
        push([=]() { vkDestroyImageView(dev, v, nullptr); });
    }

    /// Execute and clear all pending deletions. Call ONLY after the
    /// associated fence has signaled (GPU done with these resources).
    void flush() {
        std::vector<Fn> batch;
        {
            std::lock_guard lk(mu_);
            batch.swap(pending_);
        }
        for (auto& fn : batch) fn();
    }

private:
    std::mutex     mu_;
    std::vector<Fn> pending_;
};

}  // namespace plce::vk2
