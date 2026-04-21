#pragma once

#include <vulkan/vulkan.h>
#include <vma/vk_mem_alloc.h>

#include <cstdint>
#include <mutex>
#include <optional>
#include <span>
#include <vector>

#include "DeletionQueue.h"
#include "VmaResources.h"
#include "platform/renderer/IRenderPath.h"  // rp::LoadedImage

namespace plce::vk {

struct TextureSlot {
    VmaImage    image;       // VMA-owned; destructs via vmaDestroyImage
    VkImageView view  = VK_NULL_HANDLE;
    uint32_t    width = 0, height = 0;
    bool        ready = false;
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
/// repeat). Callers that would customise per-texture sampler state are
/// silently ignored; adding per-texture samplers would require a second
/// sampler binding and a selector bit in the push constant.
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
        // Completion value on the Device's shared upload timeline
        // semaphore. The upload is complete when vkGetSemaphoreCounter-
        // Value(device, upload_timeline_) >= this value.
        uint64_t        timeline_value = 0;
        VkCommandBuffer cmd          = VK_NULL_HANDLE;
        // One-shot fallback path: populated when the upload did not fit in
        // the persistent staging ring. VmaBuffer destructor runs when
        // pending_uploads_ erases the entry, so complete_upload does not
        // need an explicit vmaDestroyBuffer.
        VmaBuffer       staging;
        // Ring path: monotonic byte counters for the range reserved by
        // this upload. complete_upload uses the MIN of ring_begin across
        // all still-pending uploads to advance staging_ring_tail_ — so
        // tail can never pass a reservation whose GPU copy is still in
        // flight, even if fences signal in a different order than
        // reservations were made. ring_end == 0 means this upload took
        // the one-shot fallback.
        VkDeviceSize    ring_begin   = 0;
        VkDeviceSize    ring_end     = 0;
        int             texture_idx  = -1;
        // Orphan fields: non-empty when TextureManager::free was called
        // on texture_idx while this upload was still in flight. The slot
        // handles live here until the fence signals; complete_upload
        // destroys them instead of publishing into the bindless set.
        VmaImage        orphan_image;
        VkImageView     orphan_view  = VK_NULL_HANDLE;
    };

    void upload_texture(int idx, int w, int h, const void* pixels);
    int  ensure_default_texture();
    int  ensure_default_lightmap();

    // Reservation returned by ring_reserve: the in-buffer offset to
    // memcpy into / reference from vkCmdCopyBufferToImage, plus the
    // monotonic begin/end counters used to track the reservation so
    // complete_upload can clamp tail advancement to the oldest still-
    // in-flight begin.
    struct RingReservation {
        VkDeviceSize offset;        // byte offset inside staging_ring_buf_
        VkDeviceSize begin_counter; // store in PendingUpload::ring_begin
        VkDeviceSize end_counter;   // store in PendingUpload::ring_end
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

    void    complete_upload(PendingUpload& pu);
    void    wait_for_upload(int texture_idx);
    void    wait_all_uploads();
    // Shared drain pattern for ensure_default_* — snapshot any pending
    // upload's timeline value for `idx` under the mutex, wait on it
    // outside, then re-acquire to finalise via wait_for_upload.
    void    drain_pending_for(int idx);

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

    // Serialises the section [ring_reserve ... dev_->submit2] of
    // upload_texture and data_update so the reservation order across
    // threads matches the on-queue submit order. Without this, a second
    // worker can jump ahead of a first and the staging ring's tail-advance
    // assumption breaks (see §2.1). Also lets data_update drop its
    // wait_for_upload: any prior upload submit for the same image has
    // completed its SHADER_READ_ONLY transition on the GPU by the time
    // data_update's own command buffer runs.
    mutable std::mutex upload_submit_mutex_;

    // Persistent staging ring shared by every texture upload. One long-
    // lived host-visible VkBuffer; each upload memcpy's pixels into the
    // ring and records the byte range in its PendingUpload. When the
    // upload fence signals, complete_upload advances staging_ring_tail_
    // past that range. Uploads that don't fit in the ring (or don't fit
    // yet — the tail hasn't moved) fall back to a one-shot vmaCreateBuffer.
    static constexpr VkDeviceSize kStagingRingSize = 32ull * 1024 * 1024;
    static constexpr VkDeviceSize kStagingRingAlign = 64;  // conservative

    VmaBuffer      staging_ring_;
    std::byte*     staging_ring_map_  = nullptr;
    VkDeviceSize   staging_ring_head_ = 0;
    VkDeviceSize   staging_ring_tail_ = 0;
    mutable std::mutex staging_ring_mutex_;

    std::vector<PendingUpload> pending_uploads_;

    std::vector<TextureSlot> textures_;
    int default_tex_  = 0;
    int default_lm_   = 0;
    int bound_tex_    = 0;
    int lightmap_tex_ = 0;

    mutable std::mutex texture_mutex_;
};

}  // namespace plce::vk
