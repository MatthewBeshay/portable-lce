#include "IndirectDrawBuffer.h"

#include <vma/vk_mem_alloc.h>

#include <stdexcept>

namespace plce::vk_render {

namespace {

VkBuffer create_device_buffer(VmaAllocator allocator, VkDeviceSize size,
                              VkBufferUsageFlags usage,
                              VmaAllocation* out_alloc) {
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = size;
    bci.usage = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO;
    aci.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    VkBuffer buf = VK_NULL_HANDLE;
    if (vmaCreateBuffer(allocator, &bci, &aci, &buf, out_alloc, nullptr) !=
        VK_SUCCESS) {
        throw std::runtime_error("IndirectDrawBuffer: vmaCreateBuffer failed");
    }
    return buf;
}

}  // namespace

IndirectDrawBuffer::IndirectDrawBuffer(VmaAllocator allocator,
                                       uint32_t max_draws)
    : allocator_(allocator), max_draws_(max_draws) {
    if (max_draws_ == 0) {
        throw std::runtime_error("IndirectDrawBuffer: max_draws must be > 0");
    }

    draws_ = create_device_buffer(
        allocator_,
        VkDeviceSize(max_draws_) * kStride,
        VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        &draws_alloc_);

    count_ = create_device_buffer(
        allocator_,
        sizeof(uint32_t),
        VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        &count_alloc_);
}

IndirectDrawBuffer::~IndirectDrawBuffer() {
    if (draws_) vmaDestroyBuffer(allocator_, draws_, draws_alloc_);
    if (count_) vmaDestroyBuffer(allocator_, count_, count_alloc_);
}

void IndirectDrawBuffer::reset_count(VkCommandBuffer cmd) {
    vkCmdFillBuffer(cmd, count_, 0, sizeof(uint32_t), 0);
}

}  // namespace plce::vk_render
