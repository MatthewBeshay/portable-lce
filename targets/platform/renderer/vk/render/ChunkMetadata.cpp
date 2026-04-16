#include "ChunkMetadata.h"

#include <vma/vk_mem_alloc.h>

#include <cstring>
#include <stdexcept>

namespace plce::vk_render {

ChunkMetadata::ChunkMetadata(VkDevice /*device*/, VmaAllocator allocator,
                             uint32_t slot_count)
    : allocator_(allocator), slot_count_(slot_count) {
    if (slot_count_ == 0) {
        throw std::runtime_error("ChunkMetadata: slot_count must be > 0");
    }

    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = size_bytes();
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO;
    aci.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    VkResult r = vmaCreateBuffer(allocator_, &bci, &aci, &buffer_,
                                 &allocation_, nullptr);
    if (r != VK_SUCCESS) {
        throw std::runtime_error("ChunkMetadata: vmaCreateBuffer failed");
    }

    cpu_.resize(slot_count_);
    dirty_.resize(slot_count_, 0);
}

ChunkMetadata::~ChunkMetadata() {
    if (buffer_) vmaDestroyBuffer(allocator_, buffer_, allocation_);
}

void ChunkMetadata::update(ChunkArena::Slot slot, const ChunkSlotData& data) {
    if (!slot.valid() || slot.index >= slot_count_) return;
    std::lock_guard lk(mutex_);
    cpu_[slot.index] = data;
    dirty_[slot.index] = 1;
}

void ChunkMetadata::clear(ChunkArena::Slot slot) {
    if (!slot.valid() || slot.index >= slot_count_) return;
    std::lock_guard lk(mutex_);
    cpu_[slot.index] = {};
    dirty_[slot.index] = 1;
}

void ChunkMetadata::flush(VkCommandBuffer cmd) {
    std::lock_guard lk(mutex_);
    // Coalesce contiguous dirty runs into single vkCmdUpdateBuffer calls.
    // vkCmdUpdateBuffer requires size <= 65536 bytes so we cap each call
    // at that boundary too.
    constexpr VkDeviceSize kMaxUpdate = 65536;
    constexpr VkDeviceSize kStride    = sizeof(ChunkSlotData);
    const uint32_t kMaxRunSlots = uint32_t(kMaxUpdate / kStride);

    uint32_t i = 0;
    while (i < slot_count_) {
        if (!dirty_[i]) { ++i; continue; }
        uint32_t run_start = i;
        uint32_t run_end   = i;
        while (run_end < slot_count_ && dirty_[run_end] &&
               (run_end - run_start) < kMaxRunSlots) {
            ++run_end;
        }
        VkDeviceSize offset = VkDeviceSize(run_start) * kStride;
        VkDeviceSize bytes  = VkDeviceSize(run_end - run_start) * kStride;
        vkCmdUpdateBuffer(cmd, buffer_, offset, bytes,
                          cpu_.data() + run_start);
        for (uint32_t j = run_start; j < run_end; ++j) dirty_[j] = 0;
        i = run_end;
    }
}

}  // namespace plce::vk_render
