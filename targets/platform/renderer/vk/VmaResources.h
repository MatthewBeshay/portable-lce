#pragma once

#include <utility>

#include <vma/vk_mem_alloc.h>
#include <vulkan/vulkan.h>

namespace plce::vk {

/// Move-only RAII owner for a (VkBuffer, VmaAllocation) pair, optionally
/// carrying the mapped pointer returned by VMA. Destruction runs
/// vmaDestroyBuffer via the captured allocator handle.
///
/// Use this for buffers that have a well-defined single owner and never
/// pass through DeletionQueue (startup-time resources, persistent staging
/// ring, per-frame transient VB, etc.). Resources that need deferred
/// destruction stay as raw (VkBuffer, VmaAllocation) pairs and go through
/// DeletionQueue, which already encodes the destruction shape.
class VmaBuffer {
public:
    VmaBuffer() = default;

    explicit VmaBuffer(VmaAllocator alloc, VkBuffer buf, VmaAllocation a,
                       void* mapped = nullptr) noexcept
        : alloc_(alloc), buffer_(buf), allocation_(a), mapped_(mapped) {}

    ~VmaBuffer() noexcept { reset(); }

    VmaBuffer(const VmaBuffer&) = delete;
    VmaBuffer& operator=(const VmaBuffer&) = delete;

    VmaBuffer(VmaBuffer&& o) noexcept { swap(o); }
    VmaBuffer& operator=(VmaBuffer&& o) noexcept {
        if (this != &o) {
            reset();
            swap(o);
        }
        return *this;
    }

    [[nodiscard]] VkBuffer      handle()     const noexcept { return buffer_; }
    [[nodiscard]] VmaAllocation allocation() const noexcept { return allocation_; }
    [[nodiscard]] void*         mapped()     const noexcept { return mapped_; }
    [[nodiscard]] explicit operator bool()   const noexcept { return buffer_ != VK_NULL_HANDLE; }

    /// Destroy the owned buffer, if any, and clear to the empty state.
    void reset() noexcept {
        if (buffer_) {
            vmaDestroyBuffer(alloc_, buffer_, allocation_);
        }
        alloc_ = nullptr;
        buffer_ = VK_NULL_HANDLE;
        allocation_ = nullptr;
        mapped_ = nullptr;
    }

private:
    void swap(VmaBuffer& o) noexcept {
        std::swap(alloc_, o.alloc_);
        std::swap(buffer_, o.buffer_);
        std::swap(allocation_, o.allocation_);
        std::swap(mapped_, o.mapped_);
    }

    VmaAllocator  alloc_      = nullptr;
    VkBuffer      buffer_     = VK_NULL_HANDLE;
    VmaAllocation allocation_ = nullptr;
    void*         mapped_     = nullptr;
};

/// Move-only RAII owner for a (VkImage, VmaAllocation) pair. Destruction
/// runs vmaDestroyImage. Image views are owned separately since VMA does
/// not manage them.
class VmaImage {
public:
    VmaImage() = default;

    explicit VmaImage(VmaAllocator alloc, VkImage img, VmaAllocation a) noexcept
        : alloc_(alloc), image_(img), allocation_(a) {}

    ~VmaImage() noexcept { reset(); }

    VmaImage(const VmaImage&) = delete;
    VmaImage& operator=(const VmaImage&) = delete;

    VmaImage(VmaImage&& o) noexcept { swap(o); }
    VmaImage& operator=(VmaImage&& o) noexcept {
        if (this != &o) {
            reset();
            swap(o);
        }
        return *this;
    }

    [[nodiscard]] VkImage       handle()     const noexcept { return image_; }
    [[nodiscard]] VmaAllocation allocation() const noexcept { return allocation_; }
    [[nodiscard]] explicit operator bool()   const noexcept { return image_ != VK_NULL_HANDLE; }

    void reset() noexcept {
        if (image_) {
            vmaDestroyImage(alloc_, image_, allocation_);
        }
        alloc_ = nullptr;
        image_ = VK_NULL_HANDLE;
        allocation_ = nullptr;
    }

private:
    void swap(VmaImage& o) noexcept {
        std::swap(alloc_, o.alloc_);
        std::swap(image_, o.image_);
        std::swap(allocation_, o.allocation_);
    }

    VmaAllocator  alloc_      = nullptr;
    VkImage       image_      = VK_NULL_HANDLE;
    VmaAllocation allocation_ = nullptr;
};

}  // namespace plce::vk
