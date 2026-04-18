#pragma once

#include <functional>
#include <vector>

#include <vma/vk_mem_alloc.h>
#include <vulkan/vulkan.h>

namespace plce::vk3 {

/// Deferred destruction queue. Resources are pushed during rendering and
/// destroyed when the owning frame's fence signals (guaranteed no GPU access).
class DeletionQueue {
public:
    void push(std::function<void()>&& fn) { q_.push_back(std::move(fn)); }

    void push_buffer(VmaAllocator alloc, VkBuffer buf, VmaAllocation a) {
        push([=]() { vmaDestroyBuffer(alloc, buf, a); });
    }

    void push_image(VmaAllocator alloc, VkImage img, VmaAllocation a) {
        push([=]() { vmaDestroyImage(alloc, img, a); });
    }

    void flush() {
        for (auto& fn : q_) fn();
        q_.clear();
    }

    ~DeletionQueue() { flush(); }

private:
    std::vector<std::function<void()>> q_;
};

}  // namespace plce::vk3
