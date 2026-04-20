#pragma once

#include <vulkan/vulkan.h>
#include <vma/vk_mem_alloc.h>

#include <cstdint>
#include <mutex>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

#include "DeletionQueue.h"
#include "platform/renderer/IRenderPath.h"  // rp::LoadedImage

namespace plce::vk {

// Per-texture sampler state (kept for TextureSetParam / legacy API).
struct SamplerKey {
    VkFilter     min_filter = VK_FILTER_NEAREST;
    VkFilter     mag_filter = VK_FILTER_NEAREST;
    VkSamplerMipmapMode mip_mode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    VkSamplerAddressMode wrap_s = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    VkSamplerAddressMode wrap_t = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    bool operator==(const SamplerKey&) const = default;
};

struct TextureSlot {
    VkImage       image = VK_NULL_HANDLE;
    VmaAllocation alloc = nullptr;
    VkImageView   view  = VK_NULL_HANDLE;
    uint32_t      width = 0, height = 0;
    bool          ready = false;
    SamplerKey    sampler_key{};  // retained for set_param; ignored by bindless path
};

/// Bindless texture manager: one descriptor set holds a sampled-image array
/// of kMaxTextures slots. Binding a texture becomes writing the view into
/// slot `idx` of the array; drawing becomes indexing the array from the
/// fragment shader with `tex_id` packed into the push-constant flags.
///
/// Layout of the bindless set (owned by Renderer, populated by TextureManager):
///   binding 0: VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE[kMaxTextures]
///   binding 1: VK_DESCRIPTOR_TYPE_SAMPLER          (immutable, diffuse)
///   binding 2: VK_DESCRIPTOR_TYPE_SAMPLER          (immutable, lightmap)
///
/// Per-texture SamplerKey changes (TextureSetParam) are stored but have no
/// effect under bindless rendering — all diffuse textures share the same
/// immutable sampler (nearest, mipmap-linear, repeat) chosen to match the
/// majority of game textures.
class TextureManager {
public:
    static constexpr uint32_t kMaxTextures = 4096;

    void init(VkDevice device, VmaAllocator allocator, VkQueue queue,
              uint32_t queue_family, VkDescriptorSet bindless_set);
    void destroy(VkDevice device, VmaAllocator allocator);

    int  create();
    void free(int idx, DeletionQueue& deletions);
    void bind(int idx);
    void bind_vertex(int idx) { lightmap_tex_ = idx; }
    void data(int w, int h, const void* pixels, int level);
    void data_update(int xo, int yo, int w, int h, const void* data, int level);
    void set_param(int param, int value);

    /// Poll pending upload fences, mark completed textures ready, free staging.
    /// Called once per frame from Renderer::StartFrame.
    void poll_uploads();

    [[nodiscard]] std::optional<rp::LoadedImage> load_texture_data(const char* filename);
    [[nodiscard]] std::optional<rp::LoadedImage> load_texture_data(std::span<const uint8_t> bytes);

    int default_tex()  const { return default_tex_; }
    int default_lm()   const { return default_lm_; }
    int bound_tex()    const { return bound_tex_; }
    int lightmap_tex() const { return lightmap_tex_; }

    /// Returns the bindless slot for the currently-bound texture, falling
    /// back to default_tex_ if the bound one is not ready. Waits lazily on
    /// a pending fence if the slot exists but the image is still uploading.
    /// `textured_out` is true when the bound slot is NOT the default.
    uint32_t resolve_bound_slot(bool& textured_out);

    /// Same for the lightmap. Returns slot index (default_lm_ if unset/unready),
    /// and sets `active_out` to false if lightmap tex is 0 (no lightmap bound).
    uint32_t resolve_lightmap_slot(bool& active_out);

private:
    struct PendingUpload {
        VkFence         fence         = VK_NULL_HANDLE;
        VkCommandBuffer cmd           = VK_NULL_HANDLE;
        VkBuffer        staging_buf   = VK_NULL_HANDLE;
        VmaAllocation   staging_alloc = nullptr;
        int             texture_idx   = -1;
    };

    void upload_texture(int idx, int w, int h, const void* pixels);
    int  ensure_default_texture();
    int  ensure_default_lightmap();

    /// Write texture slot `idx`'s current view into binding 0 of the bindless set.
    void write_slot(int idx);

    VkFence acquire_fence();
    void    release_fence(VkFence);
    void    complete_upload(PendingUpload& pu);
    void    wait_for_upload(int texture_idx);
    void    wait_all_uploads();

    VkDevice     device_       = VK_NULL_HANDLE;
    VmaAllocator allocator_    = nullptr;
    VkQueue      queue_        = VK_NULL_HANDLE;

    /// Descriptor set owning the SAMPLED_IMAGE[kMaxTextures] array. Owned by
    /// Renderer; TextureManager holds a non-owning handle and writes into it.
    VkDescriptorSet bindless_set_ = VK_NULL_HANDLE;

    // Upload pool — externally synchronized via the mutex below.
    VkCommandPool      upload_pool_ = VK_NULL_HANDLE;
    mutable std::mutex upload_pool_mutex_;
    static constexpr VkDeviceSize kMaxUploadBytes = 64ull * 1024 * 1024;

    std::vector<PendingUpload> pending_uploads_;
    std::vector<VkFence>       fence_pool_;

    std::vector<TextureSlot> textures_;
    int default_tex_  = 0;
    int default_lm_   = 0;
    int bound_tex_    = 0;
    int lightmap_tex_ = 0;

    mutable std::mutex texture_mutex_;
};

}  // namespace plce::vk
