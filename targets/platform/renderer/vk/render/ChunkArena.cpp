#include "ChunkArena.h"

#include <vma/vk_mem_alloc.h>

#include <stdexcept>

namespace plce::vk_render {

ChunkArena::ChunkArena(VkDevice device, VmaAllocator allocator,
                       VkDeviceSize slot_size_bytes, VkDeviceSize total_bytes)
    : device_(device),
      allocator_(allocator),
      slot_size_bytes_(slot_size_bytes) {
    if (slot_size_bytes_ == 0) {
        throw std::runtime_error("ChunkArena: slot_size_bytes must be > 0");
    }
    slot_count_ = uint32_t((total_bytes + slot_size_bytes_ - 1) /
                           slot_size_bytes_);
    total_bytes_ = VkDeviceSize(slot_count_) * slot_size_bytes_;

    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = total_bytes_;
    bci.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO;
    aci.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    VkResult r = vmaCreateBuffer(allocator_, &bci, &aci, &buffer_,
                                 &allocation_, nullptr);
    if (r != VK_SUCCESS) {
        throw std::runtime_error("ChunkArena: vmaCreateBuffer failed");
    }

    free_list_.reserve(slot_count_);
    // Push in reverse so allocate() pops the lowest index first - mostly
    // a debugging convenience.
    for (uint32_t i = slot_count_; i-- > 0;) {
        free_list_.push_back(i);
    }
}

ChunkArena::~ChunkArena() {
    if (buffer_) {
        vmaDestroyBuffer(allocator_, buffer_, allocation_);
    }
}

ChunkArena::Slot ChunkArena::allocate() {
    std::lock_guard lk(mutex_);
    if (free_list_.empty()) return {};
    Slot s{free_list_.back()};
    free_list_.pop_back();
    return s;
}

void ChunkArena::free(Slot slot) {
    if (!slot.valid() || slot.index >= slot_count_) return;
    std::lock_guard lk(mutex_);
    free_list_.push_back(slot.index);
}

}  // namespace plce::vk_render
