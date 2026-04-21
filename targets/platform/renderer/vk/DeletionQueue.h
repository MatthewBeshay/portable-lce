#pragma once

#include <functional>
#include <utility>
#include <vector>

#include <vma/vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include "VmaResources.h"

namespace plce::vk {

/// Deferred destruction queue. Resources are pushed during rendering and
/// destroyed when the owning frame's fence signals (guaranteed no GPU access).
/// Not thread-safe — must only be accessed from the thread that owns the frame.
class DeletionQueue {
public:
    void push(std::function<void()>&& fn) {
        entries_.push_back({Tag::generic, nullptr, {}, nullptr, VK_NULL_HANDLE, nullptr, std::move(fn)});
    }

    void push_buffer(VmaAllocator alloc, VkBuffer buf, VmaAllocation a) {
        entries_.push_back({Tag::buffer, alloc, {.buffer = buf}, a, VK_NULL_HANDLE, nullptr, {}});
    }

    void push_image(VmaAllocator alloc, VkImage img, VmaAllocation a) {
        entries_.push_back({Tag::image, alloc, {.image = img}, a, VK_NULL_HANDLE, nullptr, {}});
    }

    /// Destroy an image + view pair allocated from VMA. This is the common
    /// shape for TextureManager::free — no std::function allocation.
    void push_view_image(VkDevice dev, VkImageView view,
                         VmaAllocator alloc, VkImage img, VmaAllocation a) {
        entries_.push_back({Tag::view_image, alloc, {.image = img}, a, view, dev, {}});
    }

    /// Ownership-transferring overloads for the VMA RAII wrappers. The
    /// wrapper is left empty; destruction happens at flush() time.
    void push_buffer(VmaBuffer&& buf) {
        if (!buf) return;
        auto r = buf.release();
        push_buffer(r.allocator, r.buffer, r.allocation);
    }
    void push_image(VmaImage&& img) {
        if (!img) return;
        auto r = img.release();
        push_image(r.allocator, r.image, r.allocation);
    }
    void push_view_image(VkDevice dev, VkImageView view, VmaImage&& img) {
        if (!img && !view) return;
        auto r = img.release();
        push_view_image(dev, view, r.allocator, r.image, r.allocation);
    }

    void flush() noexcept {
        for (auto& e : entries_) {
            switch (e.tag) {
                case Tag::buffer:     vmaDestroyBuffer(e.alloc, e.handle.buffer, e.vma_alloc); break;
                case Tag::image:      vmaDestroyImage(e.alloc, e.handle.image, e.vma_alloc); break;
                case Tag::view_image: {
                    if (e.view) vkDestroyImageView(e.device, e.view, nullptr);
                    if (e.handle.image) vmaDestroyImage(e.alloc, e.handle.image, e.vma_alloc);
                    break;
                }
                case Tag::generic:    if (e.fn) e.fn(); break;
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
    enum class Tag : uint8_t { buffer, image, view_image, generic };
    union Handle { VkBuffer buffer; VkImage image; };
    struct Entry {
        Tag           tag;
        VmaAllocator  alloc     = nullptr;
        Handle        handle    = {};
        VmaAllocation vma_alloc = nullptr;
        VkImageView   view      = VK_NULL_HANDLE;  // view_image only
        VkDevice      device    = nullptr;          // view_image only
        std::function<void()> fn;                   // generic only
    };
    std::vector<Entry> entries_;
};

}  // namespace plce::vk
