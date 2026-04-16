#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <mutex>
#include <vector>

#include "ChunkArena.h"

struct VmaAllocator_T;
struct VmaAllocation_T;
using VmaAllocator = VmaAllocator_T*;
using VmaAllocation = VmaAllocation_T*;

namespace plce::vk_render {

// One slot per ChunkArena slot. Layout matches std430 in
// terrain.cull.comp / terrain.vert; keep the field order in sync if you
// change the shader.
struct alignas(16) ChunkSlotData {
    float    world_pos[3];   //  0:12  chunk origin in world units
    uint32_t vert_count;     // 12:16  vertices in this slot (0 = empty)
    float    aabb_min[3];    // 16:28  world-space AABB min
    uint32_t base_vertex;    // 28:32  byte offset / vertex_stride
    float    aabb_max[3];    // 32:44  world-space AABB max
    uint32_t face_mask;      // 44:48  visible faces bitmask (FACE_*)
};
static_assert(sizeof(ChunkSlotData) == 48,
              "ChunkSlotData must match the std430 layout in shaders");

enum FaceMaskBit : uint32_t {
    FACE_DOWN  = 1u << 0,
    FACE_UP    = 1u << 1,
    FACE_NORTH = 1u << 2,
    FACE_SOUTH = 1u << 3,
    FACE_WEST  = 1u << 4,
    FACE_EAST  = 1u << 5,
};

// Stores a CPU-side mirror plus a device-local SSBO that the GPU reads.
// `update(slot, data)` patches the CPU side; `flush(cmd)` records a
// vkCmdUpdateBuffer for any dirty rows since the last flush. For Phase 4
// step 1 we use vkCmdUpdateBuffer (capped at 64 KB per call) which is
// fine for the change rate of chunk meshing.
class ChunkMetadata {
public:
    ChunkMetadata(VkDevice device, VmaAllocator allocator,
                  uint32_t slot_count);
    ~ChunkMetadata();

    ChunkMetadata(const ChunkMetadata&) = delete;
    ChunkMetadata& operator=(const ChunkMetadata&) = delete;

    void update(ChunkArena::Slot slot, const ChunkSlotData& data);
    void clear (ChunkArena::Slot slot);

    // Records vkCmdUpdateBuffer entries for everything dirty. Caller is
    // responsible for issuing the appropriate barrier afterwards.
    void flush(VkCommandBuffer cmd);

    [[nodiscard]] VkBuffer    buffer()      const noexcept { return buffer_; }
    [[nodiscard]] VkDeviceSize size_bytes() const noexcept {
        return VkDeviceSize(slot_count_) * sizeof(ChunkSlotData);
    }
    [[nodiscard]] uint32_t    slot_count()  const noexcept { return slot_count_; }

private:
    VmaAllocator   allocator_  = nullptr;
    VkBuffer       buffer_     = VK_NULL_HANDLE;
    VmaAllocation  allocation_ = nullptr;
    uint32_t       slot_count_ = 0;

    std::vector<ChunkSlotData> cpu_;
    std::vector<uint8_t>       dirty_;  // 0/1 per slot
    std::mutex                 mutex_;
};

}  // namespace plce::vk_render
