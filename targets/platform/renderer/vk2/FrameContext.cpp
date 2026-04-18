#include "FrameContext.h"

#include <stdexcept>

namespace plce::vk2 {

namespace {
void check(VkResult r, const char* w) {
    if (r != VK_SUCCESS) {
        char b[128];
        std::snprintf(b, sizeof b, "%s: %d", w, int(r));
        throw std::runtime_error(b);
    }
}
}  // namespace

void FrameContext::create(VkDevice dev, VmaAllocator alloc,
                          uint32_t queue_family) {
    // Command pool + buffer
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = queue_family;
    check(vkCreateCommandPool(dev, &pci, nullptr, &pool), "cmd pool");

    VkCommandBufferAllocateInfo ai{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    check(vkAllocateCommandBuffers(dev, &ai, &cmd), "cmd buf");

    // Sync objects
    VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    check(vkCreateSemaphore(dev, &sci, nullptr, &sem_acquired), "sem acq");
    check(vkCreateSemaphore(dev, &sci, nullptr, &sem_done), "sem done");

    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    check(vkCreateFence(dev, &fci, nullptr, &fence), "fence");

    // Transient vertex buffer (host-visible, persistently mapped)
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
    transient_mapped = static_cast<std::byte*>(info.pMappedData);
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
    // Wait for GPU to finish this frame slot
    vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX);

    // Safe to destroy resources from the previous use of this slot
    deletions.flush();

    // Reset for new recording
    vkResetFences(dev, 1, &fence);
    vkResetCommandBuffer(cmd, 0);
    transient_offset = 0;

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
}

}  // namespace plce::vk2
