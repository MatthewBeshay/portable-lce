#include "FrameContext.h"
#include "VkCheck.h"

namespace plce::vk {

void FrameContext::create(VkDevice dev, VmaAllocator alloc, uint32_t queue_family) {
    // Command pool + primary command buffer
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
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

    // Transient vertex buffer — host-visible, persistently mapped
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size  = kTransientSize;
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
}

void FrameContext::destroy(VkDevice dev, VmaAllocator alloc) {
    deletions.flush();
    if (transient_vb) vmaDestroyBuffer(alloc, transient_vb, transient_alloc);
    if (fence)        vkDestroyFence(dev, fence, nullptr);
    if (sem_done)     vkDestroySemaphore(dev, sem_done, nullptr);
    if (sem_acquired) vkDestroySemaphore(dev, sem_acquired, nullptr);
    if (pool)         vkDestroyCommandPool(dev, pool, nullptr);
}

void FrameContext::begin(VkDevice dev) {
    vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX);
    deletions.flush();
    vkResetFences(dev, 1, &fence);
    vkResetCommandBuffer(cmd, 0);
    transient_offset_ = 0;

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
}

}  // namespace plce::vk
