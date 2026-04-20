#pragma once

#include <functional>
#include <vector>

#include <vma/vk_mem_alloc.h>
#include <vulkan/vulkan.h>

namespace plce::vk3 {

/// Deferred destruction queue. Resources are pushed during rendering and
/// destroyed when the owning frame's fence signals (guaranteed no GPU access).
/// Not thread-safe — must only be accessed from the thread that owns the frame.
class DeletionQueue {
public:
    DeletionQueue() = default;
    ~DeletionQueue() noexcept { flush(); }

    DeletionQueue(const DeletionQueue&) = delete;
    DeletionQueue& operator=(const DeletionQueue&) = delete;
    DeletionQueue(DeletionQueue&&) noexcept = default;
    DeletionQueue& operator=(DeletionQueue&&) noexcept = default;

    void push(std::function<void()>&& fn) { queue_.push_back(std::move(fn)); }

    void push_buffer(VmaAllocator alloc, VkBuffer buf, VmaAllocation a) {
        push([=]() { vmaDestroyBuffer(alloc, buf, a); });
    }

    void push_image(VmaAllocator alloc, VkImage img, VmaAllocation a) {
        push([=]() { vmaDestroyImage(alloc, img, a); });
    }

    void flush() noexcept {
        for (auto& fn : queue_) fn();
        queue_.clear();
    }

private:
    std::vector<std::function<void()>> queue_;
};

}  // namespace plce::vk3
