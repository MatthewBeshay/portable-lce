#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>

struct VmaAllocator_T;
struct VmaAllocation_T;
using VmaAllocator = VmaAllocator_T*;
using VmaAllocation = VmaAllocation_T*;

namespace plce::vk_render {

// Pair of device-local buffers used by vkCmdDrawIndirectCount:
//   - draws: array of VkDrawIndirectCommand (sized for max chunks)
//   - count: single uint32 holding the visible count
//
// Both are written by the cull compute shader and consumed by the
// indirect draw call. They live in DEVICE_LOCAL memory and are reset
// each frame via vkCmdFillBuffer(count, 0).
class IndirectDrawBuffer {
public:
    IndirectDrawBuffer(VmaAllocator allocator, uint32_t max_draws);
    ~IndirectDrawBuffer();

    IndirectDrawBuffer(const IndirectDrawBuffer&) = delete;
    IndirectDrawBuffer& operator=(const IndirectDrawBuffer&) = delete;

    // Records vkCmdFillBuffer(count, 0) so the cull compute shader can
    // start its atomic-append from zero. Caller is responsible for the
    // surrounding pipeline barriers.
    void reset_count(VkCommandBuffer cmd);

    [[nodiscard]] VkBuffer draws_buffer() const noexcept { return draws_; }
    [[nodiscard]] VkBuffer count_buffer() const noexcept { return count_; }
    [[nodiscard]] uint32_t max_draws()    const noexcept { return max_draws_; }

    // Vulkan guarantees sizeof(VkDrawIndirectCommand) == 16.
    static constexpr VkDeviceSize kStride = 16;

private:
    VmaAllocator   allocator_       = nullptr;
    VkBuffer       draws_           = VK_NULL_HANDLE;
    VmaAllocation  draws_alloc_     = nullptr;
    VkBuffer       count_           = VK_NULL_HANDLE;
    VmaAllocation  count_alloc_     = nullptr;
    uint32_t       max_draws_       = 0;
};

}  // namespace plce::vk_render
