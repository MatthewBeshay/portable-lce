#pragma once

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

struct VmaAllocator_T;
struct VmaAllocation_T;
using VmaAllocator = VmaAllocator_T*;
using VmaAllocation = VmaAllocation_T*;

namespace plce::vk_render {

// ChunkArena owns one large device-local VkBuffer and hands out fixed-size
// slots for chunk vertex data. Allocations are slot indices that map to
// byte offsets via `slot_size_bytes_`.
//
// Threading: alloc/free are mutex-protected. Buffer ownership / device
// access happens on the main thread only - callers stage their writes via
// a separate StagingRing (TBD).
class ChunkArena {
public:
    struct Slot {
        uint32_t index = UINT32_MAX;
        bool valid() const noexcept { return index != UINT32_MAX; }
    };

    // total_bytes will be rounded up to a multiple of slot_size_bytes.
    ChunkArena(VkDevice device, VmaAllocator allocator,
               VkDeviceSize slot_size_bytes,
               VkDeviceSize total_bytes);
    ~ChunkArena();

    ChunkArena(const ChunkArena&) = delete;
    ChunkArena& operator=(const ChunkArena&) = delete;

    [[nodiscard]] Slot allocate();
    void free(Slot slot);

    [[nodiscard]] VkBuffer        buffer()         const noexcept { return buffer_; }
    [[nodiscard]] VkDeviceSize    slot_size()      const noexcept { return slot_size_bytes_; }
    [[nodiscard]] VkDeviceSize    capacity_bytes() const noexcept { return total_bytes_; }
    [[nodiscard]] uint32_t        slot_count()     const noexcept { return slot_count_; }

    [[nodiscard]] VkDeviceSize offset_of(Slot slot) const noexcept {
        return VkDeviceSize(slot.index) * slot_size_bytes_;
    }

private:
    VkDevice       device_     = VK_NULL_HANDLE;
    VmaAllocator   allocator_  = nullptr;
    VkBuffer       buffer_     = VK_NULL_HANDLE;
    VmaAllocation  allocation_ = nullptr;

    VkDeviceSize   slot_size_bytes_ = 0;
    VkDeviceSize   total_bytes_     = 0;
    uint32_t       slot_count_      = 0;

    std::vector<uint32_t> free_list_;  // free slot indices
    std::mutex            mutex_;
};

}  // namespace plce::vk_render
