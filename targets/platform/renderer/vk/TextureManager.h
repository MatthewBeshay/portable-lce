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
    /// back to default_tex_ if the bound one is not ready. The draw path
    /// never waits on a pending upload fence — stale binds render as the
    /// 1x1 default for a frame or two until complete_upload() publishes
    /// the real view. `textured_out` is true when the bound slot is NOT
    /// the default.
    uint32_t resolve_bound_slot(bool& textured_out);

    /// Same for the lightmap. Returns slot index (default_lm_ if unset/unready),
    /// and sets `active_out` to false if lightmap tex is 0 (no lightmap bound).
    uint32_t resolve_lightmap_slot(bool& active_out);

private:
    struct PendingUpload {
        VkFence         fence         = VK_NULL_HANDLE;
        VkCommandBuffer cmd           = VK_NULL_HANDLE;
        // One-shot fallback path: populated when the upload did not fit in
        // the persistent staging ring. Destroyed in complete_upload.
        VkBuffer        staging_buf   = VK_NULL_HANDLE;
        VmaAllocation   staging_alloc = nullptr;
        // Ring path: monotonic byte counter at which the upload's reserved
        // range ends. complete_upload advances staging_ring_tail_ to this
        // value. Zero when the one-shot fallback was used.
        VkDeviceSize    ring_end      = 0;
        int             texture_idx   = -1;
    };

    void upload_texture(int idx, int w, int h, const void* pixels);
    int  ensure_default_texture();
    int  ensure_default_lightmap();

    // Reservation returned by ring_reserve: the in-buffer offset to
    // memcpy into / reference from vkCmdCopyBufferToImage, plus the
    // monotonic end counter used to advance the tail on completion.
    struct RingReservation {
        VkDeviceSize offset;      // byte offset inside staging_ring_buf_
        VkDeviceSize end_counter; // store in PendingUpload::ring_end
    };

    /// Reserve a contiguous byte range inside the staging ring for a single
    /// upload. Returns std::nullopt if the request exceeds the ring size
    /// or the ring is currently full — callers fall back to a one-shot
    /// vmaCreateBuffer in that case.
    std::optional<RingReservation> ring_reserve(VkDeviceSize bytes);

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

    // Persistent staging ring shared by every texture upload. One long-
    // lived host-visible VkBuffer; each upload memcpy's pixels into the
    // ring and records the byte range in its PendingUpload. When the
    // upload fence signals, complete_upload advances staging_ring_tail_
    // past that range. Uploads that don't fit in the ring (or don't fit
    // yet — the tail hasn't moved) fall back to a one-shot vmaCreateBuffer.
    static constexpr VkDeviceSize kStagingRingSize = 32ull * 1024 * 1024;
    static constexpr VkDeviceSize kStagingRingAlign = 64;  // conservative

    VkBuffer       staging_ring_buf_   = VK_NULL_HANDLE;
    VmaAllocation  staging_ring_alloc_ = nullptr;
    std::byte*     staging_ring_map_   = nullptr;
    VkDeviceSize   staging_ring_head_  = 0;
    VkDeviceSize   staging_ring_tail_  = 0;
    mutable std::mutex staging_ring_mutex_;

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
