#pragma once

#include <vulkan/vulkan.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <glm/glm.hpp>
#include <memory>
#include <unordered_map>

#include "ChunkArena.h"
#include "ChunkMetadata.h"
#include "IndirectDrawBuffer.h"
#include "StagingRing.h"

struct VmaAllocator_T;
using VmaAllocator = VmaAllocator_T*;

namespace plce::vk_render {

// Owns the GPU-driven terrain pipeline. Replaces the per-chunk CBuff
// playback with: (1) one upload-on-rebuild path, (2) one compute pass
// to cull, (3) one vkCmdDrawIndirectCount per layer.
//
// Caller is the bridge layer in VulkanRenderPath - the game's chunk
// meshing routes here via a compile-time switch in Chunk::rebuild()
// (see PLCE_VK_GPU_CHUNKS).
class TerrainRenderer {
public:
    struct Config {
        // 1 GB / 1 MB = 1024 slots. Each slot holds up to ~32k verts of
        // triangulated chunk geometry. Dense edge chunks easily exceed
        // the older 8k-vert ceiling; silently dropping them showed up as
        // visible gaps in the canopy.
        VkDeviceSize arena_bytes      = 1024ull * 1024 * 1024;  // 1 GB
        VkDeviceSize slot_bytes       = 1024ull * 1024;          // 1 MB / chunk
        VkDeviceSize staging_bytes    = 128ull * 1024 * 1024;    // 128 MB ring
        uint32_t     max_visible      = 16384;                   // indirect draws
        VkFormat     color_format     = VK_FORMAT_B8G8R8A8_UNORM;
        VkFormat     depth_format     = VK_FORMAT_D32_SFLOAT;
    };

    // Identifies one chunk-layer slot, e.g. (chunk_xz, layer 0=opaque /
    // 1=alpha-test / 2=transparent).
    struct ChunkKey {
        int32_t cx, cy, cz;
        uint8_t layer;
        bool operator==(const ChunkKey&) const = default;
    };
    struct ChunkKeyHash {
        size_t operator()(const ChunkKey& k) const noexcept {
            uint64_t h = uint64_t(uint32_t(k.cx)) * 0x9E3779B185EBCA87ull;
            h ^= uint64_t(uint32_t(k.cy)) * 0xC2B2AE3D27D4EB4Full;
            h ^= uint64_t(uint32_t(k.cz)) * 0x165667B19E3779F9ull;
            h ^= uint64_t(k.layer);
            return std::hash<uint64_t>{}(h);
        }
    };

    TerrainRenderer(VkDevice device, VmaAllocator allocator,
                    uint32_t graphics_queue_family,
                    VkQueue  transfer_queue,
                    const Config& cfg = {});
    ~TerrainRenderer();

    TerrainRenderer(const TerrainRenderer&) = delete;
    TerrainRenderer& operator=(const TerrainRenderer&) = delete;

    // Called from chunk meshing threads. Copies vertex data into staging
    // (host-visible), records metadata; the next render() will move it
    // into the arena via vkCmdCopyBuffer and run the cull/draw.
    //
    // vertex_stride must match the terrain pipeline's expected layout.
    // For Phase 4 step 1 we expect the standard 32-byte format.
    void upload_chunk(const ChunkKey& key,
                      const glm::vec3& world_origin,
                      const glm::vec3& aabb_min,
                      const glm::vec3& aabb_max,
                      const void* vertex_data,
                      uint32_t vertex_count,
                      uint32_t vertex_stride);

    void destroy_chunk(const ChunkKey& key);

    // Bind / rebind the terrain atlas. Must be called at least once
    // before render() does anything. Cheap to call - just rewrites the
    // descriptor set's image binding.
    void set_atlas(VkImageView view, VkSampler sampler);

    // Bind / rebind the lightmap. Optional - if never called, the
    // shader falls back to a constant (full-bright) sample.
    void set_lightmap(VkImageView view, VkSampler sampler);

    // Records: (1) staging->arena copies queued since the last frame,
    // (2) metadata flush, (3) compute cull dispatch, (4) indirect draw.
    // Caller is responsible for issuing vkCmdBeginRendering / End around
    // the call and providing the camera MVP.
    //
    // begin_pass = true means we're already inside a render pass; if the
    // caller hasn't started one yet, set false and we'll record copies +
    // cull only (those happen outside the render pass).
    struct FogParams {
        // x = mode (0=off, 1=linear, 2=exp, 3=exp2)
        glm::vec4 params{0.0f, 0.0f, 1.0f, 1.0f};
        glm::vec4 colour{0.5f, 0.7f, 1.0f, 1.0f};
    };

    void render(VkCommandBuffer cmd,
                const glm::mat4& mvp,
                const std::array<glm::vec4, 6>& frustum_planes,
                const FogParams& fog = {},
                const glm::vec4& tint = glm::vec4(1.0f));

    // Called by VulkanRenderPath::Present once the frame fence signals
    // that the GPU has consumed up to `checkpoint`. Releases the staging
    // ring back to the writer threads.
    void release_staging(uint64_t checkpoint);

    [[nodiscard]] VkDescriptorSetLayout cull_set_layout() const noexcept {
        return cull_set_layout_;
    }
    [[nodiscard]] VkDescriptorSetLayout terrain_set_layout() const noexcept {
        return terrain_set_layout_;
    }

private:
    void create_descriptor_layouts();
    void destroy_descriptor_layouts();
    void create_descriptor_pool_and_sets();
    void destroy_descriptor_pool();
    void create_pipelines();
    void destroy_pipelines();

    VkDevice     device_   = VK_NULL_HANDLE;
    VmaAllocator allocator_ = nullptr;
    Config       cfg_{};

    std::unique_ptr<ChunkArena>          arena_;
    std::unique_ptr<ChunkMetadata>       metadata_;
    std::unique_ptr<StagingRing>         staging_;
    std::unique_ptr<IndirectDrawBuffer>  indirect_;

    VkDescriptorSetLayout cull_set_layout_    = VK_NULL_HANDLE;
    VkDescriptorSetLayout terrain_set_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout      cull_pipeline_layout_    = VK_NULL_HANDLE;
    VkPipelineLayout      terrain_pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline            cull_pipeline_    = VK_NULL_HANDLE;
    VkPipeline            terrain_pipeline_ = VK_NULL_HANDLE;

    VkDescriptorPool      desc_pool_     = VK_NULL_HANDLE;
    VkDescriptorSet       cull_set_      = VK_NULL_HANDLE;
    VkDescriptorSet       terrain_set_   = VK_NULL_HANDLE;
    bool                  atlas_set_     = false;
    bool                  lightmap_set_  = false;

    // Pending staging->arena copies queued between frames.
    struct PendingCopy {
        VkDeviceSize staging_offset;
        VkDeviceSize arena_offset;
        VkDeviceSize bytes;
        uint64_t     staging_checkpoint;
    };
    std::vector<PendingCopy> pending_copies_;
    std::mutex               pending_mutex_;

    // ChunkKey -> arena slot mapping for destroy_chunk.
    std::unordered_map<ChunkKey, ChunkArena::Slot, ChunkKeyHash> live_slots_;
    std::mutex                                                   live_slots_mutex_;

    // Per-second stats logged to stderr.
    std::atomic<uint32_t> stat_uploads_{0};
    std::atomic<uint32_t> stat_drops_oversize_{0};
    std::atomic<uint32_t> stat_drops_arena_full_{0};
    std::atomic<uint32_t> stat_drops_staging_full_{0};
    std::atomic<uint32_t> stat_renders_{0};
    std::atomic<uint32_t> stat_active_slots_{0};
    double                stat_window_start_ = 0.0;
};

}  // namespace plce::vk_render
