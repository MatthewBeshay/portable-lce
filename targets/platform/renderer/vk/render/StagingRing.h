#pragma once

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <mutex>

struct VmaAllocator_T;
struct VmaAllocation_T;
using VmaAllocator = VmaAllocator_T*;
using VmaAllocation = VmaAllocation_T*;

namespace plce::vk_render {

// StagingRing is a persistently-mapped host-visible buffer used as a ring
// allocator for staging uploads (chunk vertex data, textures, etc.). The
// ring wraps once `release_up_to(checkpoint)` confirms the GPU has
// finished consuming a previous range.
//
// Threading: alloc is mutex-protected. release_up_to should be called on
// the main thread once per frame (typically when the frame fence signals).
class StagingRing {
public:
    struct Allocation {
        std::byte*    ptr            = nullptr;
        VkDeviceSize  buffer_offset  = 0;
        VkDeviceSize  bytes          = 0;
        uint64_t      checkpoint     = 0;  // returned head position
        bool valid() const noexcept { return ptr != nullptr; }
    };

    StagingRing(VmaAllocator allocator, VkDeviceSize size_bytes);
    ~StagingRing();

    StagingRing(const StagingRing&) = delete;
    StagingRing& operator=(const StagingRing&) = delete;

    // Returns a writable range of `bytes` from the ring. Returns an
    // invalid Allocation if the ring is full (caller must wait for the
    // GPU to drain via release_up_to).
    [[nodiscard]] Allocation alloc(VkDeviceSize bytes,
                                   VkDeviceSize alignment = 16);

    // Marks all bytes [tail, checkpoint) as released - safe to overwrite.
    void release_up_to(uint64_t checkpoint);

    [[nodiscard]] VkBuffer    buffer()     const noexcept { return buffer_; }
    [[nodiscard]] VkDeviceSize size_bytes() const noexcept { return size_bytes_; }

private:
    VmaAllocator   allocator_  = nullptr;
    VkBuffer       buffer_     = VK_NULL_HANDLE;
    VmaAllocation  allocation_ = nullptr;
    std::byte*     mapped_     = nullptr;
    VkDeviceSize   size_bytes_ = 0;

    // head/tail are absolute byte counts (monotonic, never wrap). The
    // physical position in the ring is (head % size_bytes_).
    uint64_t       head_ = 0;
    uint64_t       tail_ = 0;
    std::mutex     mutex_;
};

}  // namespace plce::vk_render
