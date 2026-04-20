#pragma once

#include <functional>
#include <vector>

#include <vma/vk_mem_alloc.h>
#include <vulkan/vulkan.h>

namespace plce::vk {

/// Deferred destruction queue. Resources are pushed during rendering and
/// destroyed when the owning frame's fence signals (guaranteed no GPU access).
/// Not thread-safe — must only be accessed from the thread that owns the frame.
class DeletionQueue {
public:
    void push(std::function<void()>&& fn) {
        entries_.push_back({Tag::generic, {}, {}, {}, std::move(fn)});
    }

    void push_buffer(VmaAllocator alloc, VkBuffer buf, VmaAllocation a) {
        entries_.push_back({Tag::buffer, alloc, {.buffer = buf}, a, {}});
    }

    void push_image(VmaAllocator alloc, VkImage img, VmaAllocation a) {
        entries_.push_back({Tag::image, alloc, {.image = img}, a, {}});
    }

    void flush() noexcept {
        for (auto& e : entries_) {
            switch (e.tag) {
                case Tag::buffer:  vmaDestroyBuffer(e.alloc, e.handle.buffer, e.vma_alloc); break;
                case Tag::image:   vmaDestroyImage(e.alloc, e.handle.image, e.vma_alloc); break;
                case Tag::generic: if (e.fn) e.fn(); break;
            }
        }
        entries_.clear();
    }

    ~DeletionQueue() noexcept { flush(); }

    DeletionQueue() = default;
    DeletionQueue(const DeletionQueue&) = delete;
    DeletionQueue& operator=(const DeletionQueue&) = delete;
    DeletionQueue(DeletionQueue&&) noexcept = default;
    DeletionQueue& operator=(DeletionQueue&&) noexcept = default;

private:
    enum class Tag : uint8_t { buffer, image, generic };
    union Handle { VkBuffer buffer; VkImage image; };
    struct Entry {
        Tag           tag;
        VmaAllocator  alloc    = nullptr;
        Handle        handle   = {};
        VmaAllocation vma_alloc = nullptr;
        std::function<void()> fn;  // only used for Tag::generic
    };
    std::vector<Entry> entries_;
};

}  // namespace plce::vk
