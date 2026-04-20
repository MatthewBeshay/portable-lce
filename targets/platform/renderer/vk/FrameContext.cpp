#include "FrameContext.h"
#include "VkCheck.h"

#include <algorithm>
#include <cstdio>

namespace plce::vk {

void FrameContext::recreate_transient(VmaAllocator alloc, VkDeviceSize new_size) {
    if (transient_vb) vmaDestroyBuffer(alloc, transient_vb, transient_alloc);
    transient_vb = VK_NULL_HANDLE;
    transient_alloc = nullptr;
    transient_mapped_ = nullptr;

    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size  = new_size;
    bci.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    VmaAllocationCreateInfo vai{};
    vai.usage = VMA_MEMORY_USAGE_AUTO;
    vai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VmaAllocationInfo info{};
    check(vmaCreateBuffer(alloc, &bci, &vai, &transient_vb,
                          &transient_alloc, &info),
          "transient vb");
    transient_mapped_ = static_cast<std::byte*>(info.pMappedData);
    transient_size_ = new_size;
}

void FrameContext::create(VkDevice dev, VmaAllocator alloc, uint32_t queue_family) {
    // Command pool + primary command buffer. No RESET_COMMAND_BUFFER_BIT —
    // we reset the whole pool once per frame instead of the single CB,
    // which is cheaper in the driver and scales to additional CBs later
    // without revisiting pool flags.
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = 0;
    pci.queueFamilyIndex = queue_family;
    check(vkCreateCommandPool(dev, &pci, nullptr, &pool), "cmd pool");

    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    check(vkAllocateCommandBuffers(dev, &ai, &cmd), "cmd buf");

    // Sync objects — fence starts signaled so first wait succeeds
    VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    check(vkCreateSemaphore(dev, &sci, nullptr, &sem_acquired), "sem acq");
    check(vkCreateSemaphore(dev, &sci, nullptr, &sem_done), "sem done");

    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    check(vkCreateFence(dev, &fci, nullptr, &fence), "fence");

    // Transient vertex buffer — host-visible, persistently mapped. Grows on demand.
    recreate_transient(alloc, kInitialTransientSize);
}

void FrameContext::reset_acquire_semaphore(VkDevice dev) {
    if (sem_acquired) vkDestroySemaphore(dev, sem_acquired, nullptr);
    VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    check(vkCreateSemaphore(dev, &sci, nullptr, &sem_acquired), "sem acq (reset)");
}

void FrameContext::destroy(VkDevice dev, VmaAllocator alloc) {
    deletions.flush();
    if (transient_vb) vmaDestroyBuffer(alloc, transient_vb, transient_alloc);
    if (fence)        vkDestroyFence(dev, fence, nullptr);
    if (sem_done)     vkDestroySemaphore(dev, sem_done, nullptr);
    if (sem_acquired) vkDestroySemaphore(dev, sem_acquired, nullptr);
    if (pool)         vkDestroyCommandPool(dev, pool, nullptr);
}

void FrameContext::begin(VkDevice dev, VmaAllocator alloc) {
    vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX);
    deletions.flush();

    // Grow the transient buffer if a prior frame overflowed. Safe here because
    // the fence has signaled — no GPU work is still reading from the old buffer.
    if (pending_grow_bytes_ > 0) {
        VkDeviceSize new_size = transient_size_ * 2;
        while (new_size < pending_grow_bytes_) new_size *= 2;
        new_size = std::min(new_size, kMaxTransientSize);
        if (new_size > transient_size_) {
            std::fprintf(stderr, "[vk] transient VB grew from %llu to %llu bytes\n",
                         (unsigned long long)transient_size_,
                         (unsigned long long)new_size);
            recreate_transient(alloc, new_size);
        }
        pending_grow_bytes_ = 0;
    }

    vkResetFences(dev, 1, &fence);
    vkResetCommandPool(dev, pool, 0);
    transient_offset_ = 0;

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
}

}  // namespace plce::vk
