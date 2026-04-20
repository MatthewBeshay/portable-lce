#pragma once

#include <vulkan/vulkan.h>
#include <vma/vk_mem_alloc.h>

#include <cstdint>
#include <mutex>
#include <optional>
#include <span>
#include <vector>

#include "DeletionQueue.h"
#include "platform/renderer/IRenderPath.h"  // rp::LoadedImage

namespace plce::vk {

struct TextureSlot {
    VkImage       image = VK_NULL_HANDLE;
    VmaAllocation alloc = nullptr;
    VkImageView   view  = VK_NULL_HANDLE;
    uint32_t      width = 0, height = 0;
    bool          ready = false;
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
/// Per-texture sampler variation is NOT supported: every diffuse texture uses
/// the immutable diffuse sampler baked into the layout (nearest, mipmap-linear,
/// repeat). TextureSetParam is therefore a no-op with a diagnostic log; adding
/// per-texture samplers would require a second sampler binding and a selector
/// bit in the push constant.
class Device;  // forward decl

class TextureManager {
public:
    static constexpr uint32_t kMaxTextures = 4096;

    /// The Device reference is stored for queue submission through
    /// Device::submit2, which guards vkQueueSubmit2 against concurrent
    /// main-thread present calls.
    void init(const Device& dev, VkDescriptorSet bindless_set);
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
    /// Write an explicit view into slot `idx` (used by free() to redirect
    /// freed slots at the 1×1 default texture before deleting the old view).
    void write_slot_with_view(int idx, VkImageView view);

    VkFence acquire_fence();
    void    release_fence(VkFence);
    void    complete_upload(PendingUpload& pu);
    void    wait_for_upload(int texture_idx);
    void    wait_all_uploads();

    const Device* dev_       = nullptr;  // non-owning; guarded submits
    VkDevice      device_    = VK_NULL_HANDLE;
    VmaAllocator  allocator_ = nullptr;

    /// Descriptor set owning the SAMPLED_IMAGE[kMaxTextures] array. Owned by
    /// Renderer; TextureManager holds a non-owning handle and writes into it.
    VkDescriptorSet bindless_set_ = VK_NULL_HANDLE;

    // Upload pool — externally synchronized via the mutex below.
    VkCommandPool      upload_pool_ = VK_NULL_HANDLE;
    mutable std::mutex upload_pool_mutex_;
    static constexpr VkDeviceSize kMaxUploadBytes = 64ull * 1024 * 1024;

    std::vector<PendingUpload> pending_uploads_;
    std::vector<VkFence>       fence_pool_;
    mutable std::mutex         fence_pool_mutex_;

    std::vector<TextureSlot> textures_;
    int default_tex_  = 0;
    int default_lm_   = 0;
    int bound_tex_    = 0;
    int lightmap_tex_ = 0;

    mutable std::mutex texture_mutex_;
};

}  // namespace plce::vk
