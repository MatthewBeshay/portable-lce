#include "StagingRing.h"

#include <vma/vk_mem_alloc.h>

#include <stdexcept>

namespace plce::vk_render {

namespace {

VkDeviceSize align_up(VkDeviceSize value, VkDeviceSize alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

}  // namespace

StagingRing::StagingRing(VmaAllocator allocator, VkDeviceSize size_bytes)
    : allocator_(allocator), size_bytes_(size_bytes) {
    if (size_bytes_ == 0) {
        throw std::runtime_error("StagingRing: size must be > 0");
    }

    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = size_bytes_;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO;
    aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VmaAllocationInfo info{};
    VkResult r = vmaCreateBuffer(allocator_, &bci, &aci, &buffer_,
                                 &allocation_, &info);
    if (r != VK_SUCCESS) {
        throw std::runtime_error("StagingRing: vmaCreateBuffer failed");
    }
    mapped_ = static_cast<std::byte*>(info.pMappedData);
}

StagingRing::~StagingRing() {
    if (buffer_) {
        vmaDestroyBuffer(allocator_, buffer_, allocation_);
    }
}

StagingRing::Allocation StagingRing::alloc(VkDeviceSize bytes,
                                           VkDeviceSize alignment) {
    if (bytes == 0 || bytes > size_bytes_) return {};
    std::lock_guard lk(mutex_);

    uint64_t head = align_up(head_, alignment);

    // Available space in front of head, before reaching tail+size.
    uint64_t in_flight = head - tail_;
    uint64_t free_bytes = (in_flight >= size_bytes_) ? 0 : size_bytes_ - in_flight;
    if (free_bytes < bytes) return {};

    // If the contiguous tail of the ring can't fit `bytes`, skip ahead so
    // the allocation lives entirely in one half. Costs us a small amount of
    // wasted space but keeps the buffer-offset math simple for callers.
    VkDeviceSize ring_pos = head % size_bytes_;
    if (ring_pos + bytes > size_bytes_) {
        // Pad to the end of the ring.
        head += size_bytes_ - ring_pos;
        in_flight = head - tail_;
        free_bytes = (in_flight >= size_bytes_) ? 0 : size_bytes_ - in_flight;
        if (free_bytes < bytes) return {};
        ring_pos = 0;
    }

    Allocation a;
    a.ptr           = mapped_ + ring_pos;
    a.buffer_offset = ring_pos;
    a.bytes         = bytes;
    a.checkpoint    = head + bytes;
    head_ = a.checkpoint;
    return a;
}

void StagingRing::release_up_to(uint64_t checkpoint) {
    std::lock_guard lk(mutex_);
    if (checkpoint > tail_) tail_ = checkpoint;
}

}  // namespace plce::vk_render
