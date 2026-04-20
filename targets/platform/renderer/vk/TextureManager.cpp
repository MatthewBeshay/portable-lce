#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "TextureManager.h"
#include "Device.h"
#include "VkCheck.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace plce::vk {

void TextureManager::init(const Device& dev, VkDescriptorSet bindless_set) {
    dev_          = &dev;
    device_       = dev.handle();
    allocator_    = dev.allocator();
    bindless_set_ = bindless_set;

    textures_.reserve(256);
    pending_uploads_.reserve(16);
    fence_pool_.reserve(16);

    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
                VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = dev.queue_family();
    check(vkCreateCommandPool(device_, &pci, nullptr, &upload_pool_), "upload pool");

    // Persistent staging ring. One host-visible, persistently-mapped buffer
    // used by every texture upload that fits. Bump-allocated via
    // ring_reserve(); reclaimed in complete_upload().
    VkBufferCreateInfo ring_bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    ring_bi.size  = kStagingRingSize;
    ring_bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VmaAllocationCreateInfo ring_ai{};
    ring_ai.usage = VMA_MEMORY_USAGE_AUTO;
    ring_ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                    VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VmaAllocationInfo ring_info{};
    VkBuffer      ring_buf   = VK_NULL_HANDLE;
    VmaAllocation ring_alloc = nullptr;
    check(vmaCreateBuffer(allocator_, &ring_bi, &ring_ai, &ring_buf,
                          &ring_alloc, &ring_info),
          "staging ring");
    staging_ring_     = VmaBuffer(allocator_, ring_buf, ring_alloc,
                                  ring_info.pMappedData);
    staging_ring_map_ = static_cast<std::byte*>(ring_info.pMappedData);

    default_tex_ = ensure_default_texture();
    default_lm_  = ensure_default_lightmap();
    bound_tex_   = default_tex_;
}

void TextureManager::destroy(VkDevice device, VmaAllocator allocator) {
    wait_all_uploads();

    staging_ring_.reset();
    staging_ring_map_ = nullptr;

    if (upload_pool_) vkDestroyCommandPool(device, upload_pool_, nullptr);
    for (VkFence f : fence_pool_) vkDestroyFence(device, f, nullptr);
    fence_pool_.clear();

    for (auto& t : textures_) {
        if (t.view)  vkDestroyImageView(device, t.view, nullptr);
        if (t.image) vmaDestroyImage(allocator, t.image, t.alloc);
    }
}

// ===================================================================
// STAGING RING
// ===================================================================

std::optional<TextureManager::RingReservation>
TextureManager::ring_reserve(VkDeviceSize bytes) {
    const VkDeviceSize aligned =
        (bytes + kStagingRingAlign - 1) & ~(kStagingRingAlign - 1);
    if (aligned == 0 || aligned > kStagingRingSize) return std::nullopt;

    std::lock_guard lk(staging_ring_mutex_);

    // head_ and tail_ are monotonic byte counters. In-buffer offset is
    // head_ mod ring_size. The number of bytes currently reserved by
    // in-flight uploads is (head_ - tail_); the ring is full when that
    // equals kStagingRingSize.
    VkDeviceSize in_flight = staging_ring_head_ - staging_ring_tail_;
    if (in_flight + aligned > kStagingRingSize) return std::nullopt;

    VkDeviceSize off = staging_ring_head_ % kStagingRingSize;
    if (off + aligned > kStagingRingSize) {
        // Request would straddle the ring boundary. Pad the monotonic
        // head up to the next multiple of ring_size so the upload sits
        // contiguously at offset 0, and re-check space.
        VkDeviceSize pad = kStagingRingSize - off;
        if (in_flight + pad + aligned > kStagingRingSize) return std::nullopt;
        staging_ring_head_ += pad;
        off = 0;
    }

    const VkDeviceSize begin_counter = staging_ring_head_;
    staging_ring_head_ += aligned;
    return RingReservation{off, begin_counter, staging_ring_head_};
}

// ===================================================================
// FENCE POOL
// ===================================================================

VkFence TextureManager::acquire_fence() {
    // Dedicated mutex: fence_pool_ lives outside the texture state, and
    // acquire_fence is called from worker-thread upload paths without
    // texture_mutex_ held while release_fence runs under texture_mutex_.
    {
        std::lock_guard lk(fence_pool_mutex_);
        if (!fence_pool_.empty()) {
            VkFence f = fence_pool_.back();
            fence_pool_.pop_back();
            return f;
        }
    }
    // Slow path (pool empty) — create a fresh fence outside the lock.
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence f = VK_NULL_HANDLE;
    check(vkCreateFence(device_, &fci, nullptr, &f), "upload fence");
    return f;
}

void TextureManager::release_fence(VkFence f) {
    std::lock_guard lk(fence_pool_mutex_);
    fence_pool_.push_back(f);
}

void TextureManager::complete_upload(PendingUpload& pu) {
    textures_[pu.texture_idx].ready = true;
    // Publish the image view into the bindless array now that it's safe to sample.
    write_slot(pu.texture_idx);

    if (pu.staging_buf) {
        // One-shot fallback path — destroy the private staging buffer.
        vmaDestroyBuffer(allocator_, pu.staging_buf, pu.staging_alloc);
    } else if (pu.ring_end != 0) {
        // Ring path — advance the tail. pu is still live in
        // pending_uploads_ at this point (poll_uploads / wait_for_upload
        // erase it only after complete_upload returns), so scan every
        // OTHER entry and clamp tail to the minimum ring_begin among
        // them. This ensures the tail never passes the start of a
        // reservation whose GPU copy is still in flight — the fence
        // signal order on the graphics queue can differ from the
        // reservation order if two workers submit between their own
        // ring_reserve and submit2 calls.
        VkDeviceSize new_tail = staging_ring_head_;
        for (const auto& other : pending_uploads_) {
            if (&other == &pu) continue;
            if (other.ring_end == 0) continue;  // one-shot fallback entry
            if (other.ring_begin < new_tail) new_tail = other.ring_begin;
        }
        std::lock_guard lk(staging_ring_mutex_);
        if (new_tail > staging_ring_tail_)
            staging_ring_tail_ = new_tail;
    }

    {
        std::lock_guard pool_lk(upload_pool_mutex_);
        vkFreeCommandBuffers(device_, upload_pool_, 1, &pu.cmd);
    }
    vkResetFences(device_, 1, &pu.fence);
    release_fence(pu.fence);
}

void TextureManager::poll_uploads() {
    std::lock_guard lk(texture_mutex_);
    for (auto it = pending_uploads_.begin(); it != pending_uploads_.end(); ) {
        if (vkGetFenceStatus(device_, it->fence) == VK_SUCCESS) {
            complete_upload(*it);
            it = pending_uploads_.erase(it);
        } else {
            ++it;
        }
    }
}

void TextureManager::wait_for_upload(int texture_idx) {
    // Caller holds texture_mutex_.
    for (auto it = pending_uploads_.begin(); it != pending_uploads_.end(); ++it) {
        if (it->texture_idx == texture_idx) {
            vkWaitForFences(device_, 1, &it->fence, VK_TRUE, UINT64_MAX);
            complete_upload(*it);
            pending_uploads_.erase(it);
            return;
        }
    }
}

void TextureManager::wait_all_uploads() {
    std::lock_guard lk(texture_mutex_);
    for (auto& pu : pending_uploads_) {
        vkWaitForFences(device_, 1, &pu.fence, VK_TRUE, UINT64_MAX);
        complete_upload(pu);
    }
    pending_uploads_.clear();
}

// ===================================================================
// BINDLESS SLOT WRITE
// ===================================================================

void TextureManager::write_slot(int idx) {
    // Caller holds texture_mutex_.
    if (bindless_set_ == VK_NULL_HANDLE) return;
    if (idx < 0 || size_t(idx) >= textures_.size()) return;
    write_slot_with_view(idx, textures_[idx].view);
}

void TextureManager::write_slot_with_view(int idx, VkImageView view) {
    // Caller holds texture_mutex_.
    if (bindless_set_ == VK_NULL_HANDLE || view == VK_NULL_HANDLE) return;

    VkDescriptorImageInfo dii{};
    dii.sampler     = VK_NULL_HANDLE;  // separate sampler binding
    dii.imageView   = view;
    dii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet wd{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    wd.dstSet          = bindless_set_;
    wd.dstBinding      = 0;
    wd.dstArrayElement = uint32_t(idx);
    wd.descriptorCount = 1;
    wd.descriptorType  = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    wd.pImageInfo      = &dii;
    vkUpdateDescriptorSets(device_, 1, &wd, 0, nullptr);
}

// ===================================================================
// RESOLVE
// ===================================================================

uint32_t TextureManager::resolve_bound_slot(bool& textured_out) {
    // Never wait on the render thread. If the bound texture's upload is
    // still in flight, fall back to the 1x1 default — the bindless array
    // snaps to the real view on the frame after the upload fence signals
    // (complete_upload -> write_slot). Prior behaviour called
    // vkWaitForFences(UINT64_MAX) here which stalled every draw behind a
    // pending upload.
    std::lock_guard lk(texture_mutex_);
    int idx = bound_tex_;
    if (idx <= 0 || size_t(idx) >= textures_.size() || !textures_[idx].ready) {
        textured_out = false;
        return uint32_t(default_tex_);
    }
    textured_out = (idx != default_tex_);
    return uint32_t(idx);
}

uint32_t TextureManager::resolve_lightmap_slot(bool& active_out) {
    // See resolve_bound_slot: no render-thread fence wait.
    std::lock_guard lk(texture_mutex_);
    int idx = lightmap_tex_;
    if (idx <= 0 || size_t(idx) >= textures_.size() || !textures_[idx].ready) {
        active_out = false;
        return uint32_t(default_lm_);
    }
    active_out = true;
    return uint32_t(idx);
}

// ===================================================================
// TEXTURES (thin legacy API)
// ===================================================================

int TextureManager::create() {
    std::lock_guard lk(texture_mutex_);
    const int idx = int(textures_.size());
    if (uint32_t(idx) >= kMaxTextures) {
        throw std::runtime_error(
            "vk::TextureManager::create: texture count exceeds bindless array size "
            "(kMaxTextures=4096). Raise kMaxTextures and the descriptor set layout "
            "in lockstep if the game needs more textures.");
    }
    textures_.emplace_back();
    return idx;
}

void TextureManager::free(int idx, DeletionQueue& deletions) {
    std::lock_guard lk(texture_mutex_);
    if (idx <= 0 || size_t(idx) >= textures_.size()) return;
    if (idx == default_tex_) return;
    wait_for_upload(idx);
    auto& t = textures_[idx];
    // Redirect the bindless slot at `idx` to the 1×1 default view before we
    // queue the real view for deletion. Without this, the slot holds a
    // handle that's about to be destroyed; if a later frame indexes this
    // slot (e.g. via a stale tex_id in a display list) the driver samples
    // a freed image view. PARTIALLY_BOUND only protects against never-
    // sampled slots, not stale handles.
    if (default_tex_ > 0 && size_t(default_tex_) < textures_.size() &&
        textures_[default_tex_].view) {
        write_slot_with_view(idx, textures_[default_tex_].view);
    }
    if (t.view || t.image)
        deletions.push_view_image(device_, t.view, allocator_, t.image, t.alloc);
    t = {};
}

void TextureManager::bind(int idx) {
    std::lock_guard lk(texture_mutex_);
    if (idx < 0) { bound_tex_ = default_tex_; return; }
    if (uint32_t(idx) >= kMaxTextures) {
        throw std::runtime_error(
            "vk::TextureManager::bind: texture index exceeds bindless array "
            "size (kMaxTextures=4096). Caller is binding an id that was never "
            "issued by create().");
    }
    if (size_t(idx) >= textures_.size()) textures_.resize(idx + 1);
    bound_tex_ = idx;
}

void TextureManager::data(int w, int h, const void* pixels, int level) {
    if (level != 0 || !pixels) return;
    int idx;
    {
        std::lock_guard lk(texture_mutex_);
        idx = bound_tex_;
        if (idx <= 0) return;
        if (uint32_t(idx) >= kMaxTextures) {
            throw std::runtime_error(
                "vk::TextureManager::data: bound texture index exceeds "
                "bindless array size (kMaxTextures=4096).");
        }
        if (size_t(idx) >= textures_.size()) textures_.resize(idx + 1);
    }

    upload_texture(idx, w, h, pixels);
}

void TextureManager::data_update(int xo, int yo, int w, int h, const void* data, int level) {
    if (level != 0 || !data) return;
    int idx;
    uint32_t tw = 0, th = 0;
    VkImage image = VK_NULL_HANDLE;
    {
        std::lock_guard lk(texture_mutex_);
        idx = bound_tex_;
        if (idx <= 0 || size_t(idx) >= textures_.size()) return;
        wait_for_upload(idx);
        if (!textures_[idx].ready) return;
        tw = textures_[idx].width;
        th = textures_[idx].height;
        image = textures_[idx].image;
    }

    if (xo == 0 && yo == 0 && uint32_t(w) == tw && uint32_t(h) == th) {
        upload_texture(idx, w, h, data);
        return;
    }
    if (xo < 0 || yo < 0 || uint32_t(xo + w) > tw || uint32_t(yo + h) > th) return;

    VkDeviceSize bytes = VkDeviceSize(w) * h * 4;
    if (bytes > kMaxUploadBytes) return;

    // Source: ring first, fall back to a one-shot vmaCreateBuffer if the
    // update doesn't fit.
    std::optional<RingReservation> res = ring_reserve(bytes);
    std::byte*    src_map       = nullptr;
    VkBuffer      src_buf       = VK_NULL_HANDLE;
    VkDeviceSize  src_offset    = 0;
    VkDeviceSize  ring_begin    = 0;
    VkDeviceSize  ring_end      = 0;
    VkBuffer      oneshot_buf   = VK_NULL_HANDLE;
    VmaAllocation oneshot_alloc = nullptr;
    if (res) {
        src_map    = staging_ring_map_ + res->offset;
        src_buf    = staging_ring_.handle();
        src_offset = res->offset;
        ring_begin = res->begin_counter;
        ring_end   = res->end_counter;
    } else {
        VkBufferCreateInfo stg_bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        stg_bi.size = bytes; stg_bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        VmaAllocationCreateInfo stg_ai{};
        stg_ai.usage = VMA_MEMORY_USAGE_AUTO;
        stg_ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                       VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VmaAllocationInfo stg_info{};
        check(vmaCreateBuffer(allocator_, &stg_bi, &stg_ai, &oneshot_buf,
                              &oneshot_alloc, &stg_info),
              "tex update staging (fallback)");
        src_map = static_cast<std::byte*>(stg_info.pMappedData);
        src_buf = oneshot_buf;
    }
    std::memcpy(src_map, data, bytes);

    VkCommandBuffer cmd;
    {
        std::lock_guard pool_lk(upload_pool_mutex_);
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = upload_pool_; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1;
        check(vkAllocateCommandBuffers(device_, &cai, &cmd), "tex update cmd");
        VkCommandBufferBeginInfo bbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &bbi);

        VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        b.srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        b.srcAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        b.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        b.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        b.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b.image = image;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dep.imageMemoryBarrierCount = 1; dep.pImageMemoryBarriers = &b;
        vkCmdPipelineBarrier2(cmd, &dep);

        VkBufferImageCopy rgn{};
        rgn.bufferOffset = src_offset;
        rgn.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        rgn.imageOffset = {xo, yo, 0};
        rgn.imageExtent = {uint32_t(w), uint32_t(h), 1};
        vkCmdCopyBufferToImage(cmd, src_buf, image,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &rgn);

        b.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        b.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        b.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        b.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        vkCmdPipelineBarrier2(cmd, &dep);

        vkEndCommandBuffer(cmd);
    }

    VkFence fence = acquire_fence();
    VkCommandBufferSubmitInfo csi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    csi.commandBuffer = cmd;
    VkSubmitInfo2 sub{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    sub.commandBufferInfoCount = 1; sub.pCommandBufferInfos = &csi;
    check(dev_->submit2(1, &sub, fence), "tex update submit");

    std::lock_guard lk(texture_mutex_);
    pending_uploads_.push_back({fence, cmd, oneshot_buf, oneshot_alloc,
                                ring_begin, ring_end, idx});
    textures_[idx].ready = false;
}

void TextureManager::set_param(int /*param*/, int /*value*/) {
    // No-op under the bindless layout: diffuse textures all share the
    // immutable diffuse sampler baked into the descriptor set layout
    // (Renderer.cpp). Per-texture filter/wrap variations are unsupported
    // by design. If a caller needs them, add a second sampler binding
    // and a selector bit in the push constant — not a silent store.
}

int TextureManager::ensure_default_texture() {
    int idx = create();
    uint32_t pixel = 0xFFFFFFFF;
    { std::lock_guard lk(texture_mutex_); bound_tex_ = idx; }
    upload_texture(idx, 1, 1, &pixel);
    { std::lock_guard lk(texture_mutex_); wait_for_upload(idx); }
    return idx;
}

int TextureManager::ensure_default_lightmap() {
    int idx = create();
    uint32_t pixel = 0xFFFFFFFF;
    { std::lock_guard lk(texture_mutex_); bound_tex_ = idx; }
    upload_texture(idx, 1, 1, &pixel);
    { std::lock_guard lk(texture_mutex_); wait_for_upload(idx); }
    return idx;
}

void TextureManager::upload_texture(int idx, int w, int h, const void* pixels) {
    // Snapshot the slot under the lock, then drop it for the long image
    // creation path. `reuse` tells the rest of the function whether we can
    // keep the existing VkImage/VkImageView or must allocate fresh ones.
    bool reuse;
    VkImage old_image = VK_NULL_HANDLE;
    VmaAllocation old_alloc = nullptr;
    VkImageView   old_view  = VK_NULL_HANDLE;
    {
        std::lock_guard lk(texture_mutex_);
        wait_for_upload(idx);
        TextureSlot& t = textures_[idx];
        reuse = t.ready && t.width == uint32_t(w) && t.height == uint32_t(h);
        if (t.ready && !reuse) {
            old_image = t.image;
            old_alloc = t.alloc;
            old_view  = t.view;
            t = {};
        }
        t.width = uint32_t(w);
        t.height = uint32_t(h);
    }
    // Destroy the old image handles outside the lock. Safe because
    // wait_for_upload above guarantees no fence still references them.
    if (old_view)  vkDestroyImageView(device_, old_view, nullptr);
    if (old_image) vmaDestroyImage(allocator_, old_image, old_alloc);

    uint32_t mips = 1;
    { uint32_t d = std::max(uint32_t(w), uint32_t(h)); while (d > 1) { d >>= 1; ++mips; } }

    VkImage       new_image = VK_NULL_HANDLE;
    VmaAllocation new_alloc = nullptr;
    VkImageView   new_view  = VK_NULL_HANDLE;
    if (!reuse) {
        VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format    = VK_FORMAT_R8G8B8A8_UNORM;
        ici.extent    = {uint32_t(w), uint32_t(h), 1};
        ici.mipLevels = mips; ici.arrayLayers = 1;
        ici.samples   = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling    = VK_IMAGE_TILING_OPTIMAL;
        ici.usage     = VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                        VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                        VK_IMAGE_USAGE_SAMPLED_BIT;
        VmaAllocationCreateInfo ai{}; ai.usage = VMA_MEMORY_USAGE_AUTO;
        check(vmaCreateImage(allocator_, &ici, &ai, &new_image, &new_alloc, nullptr),
              "texture image");

        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image = new_image; vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = VK_FORMAT_R8G8B8A8_UNORM;
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mips, 0, 1};
        check(vkCreateImageView(device_, &vci, nullptr, &new_view), "texture view");

        std::lock_guard lk(texture_mutex_);
        TextureSlot& t = textures_[idx];
        t.image = new_image;
        t.alloc = new_alloc;
        t.view  = new_view;
    }
    // Capture the image handle for the command buffer recording below. Under
    // reuse this is the existing image; under the !reuse path it's the one we
    // just created and stored.
    VkImage upload_image;
    {
        std::lock_guard lk(texture_mutex_);
        upload_image = textures_[idx].image;
    }

    VkDeviceSize bytes = VkDeviceSize(w) * h * 4;
    if (bytes > kMaxUploadBytes) return;

    // Source: ring first, fall back to a one-shot vmaCreateBuffer if the
    // upload doesn't fit or the ring is momentarily full.
    std::optional<RingReservation> res = ring_reserve(bytes);
    std::byte*    src_map       = nullptr;
    VkBuffer      src_buf       = VK_NULL_HANDLE;
    VkDeviceSize  src_offset    = 0;
    VkDeviceSize  ring_begin    = 0;
    VkDeviceSize  ring_end      = 0;
    VkBuffer      oneshot_buf   = VK_NULL_HANDLE;
    VmaAllocation oneshot_alloc = nullptr;
    if (res) {
        src_map    = staging_ring_map_ + res->offset;
        src_buf    = staging_ring_.handle();
        src_offset = res->offset;
        ring_begin = res->begin_counter;
        ring_end   = res->end_counter;
    } else {
        VkBufferCreateInfo stg_bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        stg_bi.size = bytes; stg_bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        VmaAllocationCreateInfo stg_ai{};
        stg_ai.usage = VMA_MEMORY_USAGE_AUTO;
        stg_ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                       VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VmaAllocationInfo stg_info{};
        check(vmaCreateBuffer(allocator_, &stg_bi, &stg_ai, &oneshot_buf,
                              &oneshot_alloc, &stg_info),
              "texture staging (fallback)");
        src_map = static_cast<std::byte*>(stg_info.pMappedData);
        src_buf = oneshot_buf;
    }
    std::memcpy(src_map, pixels, bytes);

    VkCommandBuffer cmd;
    {
        std::lock_guard pool_lk(upload_pool_mutex_);
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = upload_pool_; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 1;
        check(vkAllocateCommandBuffers(device_, &cai, &cmd), "texture upload cmd");
        VkCommandBufferBeginInfo bbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &bbi);

        // UNDEFINED -> TRANSFER_DST (no prior producer)
        {
            VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
            b.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
            b.srcAccessMask = 0;
            b.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
            b.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            b.image = upload_image;
            b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mips, 0, 1};
            VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            dep.imageMemoryBarrierCount = 1; dep.pImageMemoryBarriers = &b;
            vkCmdPipelineBarrier2(cmd, &dep);
        }
        VkBufferImageCopy rgn{};
        rgn.bufferOffset = src_offset;
        rgn.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        rgn.imageExtent = {uint32_t(w), uint32_t(h), 1};
        vkCmdCopyBufferToImage(cmd, src_buf, upload_image,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &rgn);

        // Mip generation
        int32_t mw = w, mh = h;
        for (uint32_t i = 1; i < mips; ++i) {
            VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
            b.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
            b.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            b.dstStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT;
            b.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
            b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            b.image = upload_image;
            b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, i-1, 1, 0, 1};
            VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            dep.imageMemoryBarrierCount = 1; dep.pImageMemoryBarriers = &b;
            vkCmdPipelineBarrier2(cmd, &dep);

            int32_t nw = std::max(mw/2, 1), nh = std::max(mh/2, 1);
            VkImageBlit blit{};
            blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i-1, 0, 1};
            blit.srcOffsets[1] = {mw, mh, 1};
            blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i, 0, 1};
            blit.dstOffsets[1] = {nw, nh, 1};
            vkCmdBlitImage(cmd, upload_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           upload_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           1, &blit, VK_FILTER_LINEAR);
            mw = nw; mh = nh;
        }

        // All mips -> SHADER_READ_ONLY
        {
            VkImageMemoryBarrier2 ends[2]{}; uint32_t n = 0;
            if (mips > 1) {
                ends[n].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                ends[n].srcStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT;
                ends[n].srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
                ends[n].dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
                ends[n].dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
                ends[n].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                ends[n].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                ends[n].image = upload_image;
                ends[n].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mips-1, 0, 1};
                ++n;
            }
            ends[n].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            ends[n].srcStageMask = (mips>1) ? VK_PIPELINE_STAGE_2_BLIT_BIT : VK_PIPELINE_STAGE_2_COPY_BIT;
            ends[n].srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            ends[n].dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
            ends[n].dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
            ends[n].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            ends[n].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            ends[n].image = upload_image;
            ends[n].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, mips-1, 1, 0, 1};
            ++n;
            VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            dep.imageMemoryBarrierCount = n; dep.pImageMemoryBarriers = ends;
            vkCmdPipelineBarrier2(cmd, &dep);
        }
        vkEndCommandBuffer(cmd);
    }

    VkFence fence = acquire_fence();
    VkCommandBufferSubmitInfo csi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    csi.commandBuffer = cmd;
    VkSubmitInfo2 sub{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    sub.commandBufferInfoCount = 1; sub.pCommandBufferInfos = &csi;
    check(dev_->submit2(1, &sub, fence), "texture upload submit");

    std::lock_guard lk(texture_mutex_);
    pending_uploads_.push_back({fence, cmd, oneshot_buf, oneshot_alloc,
                                ring_begin, ring_end, idx});
}

namespace {
rp::LoadedImage pack_argb(unsigned char* px, int w, int h) {
    rp::LoadedImage out;
    out.width  = w;
    out.height = h;
    out.argb_pixels.resize(size_t(w) * size_t(h));
    for (int i = 0; i < w * h; ++i) {
        unsigned char r = px[i * 4 + 0];
        unsigned char g = px[i * 4 + 1];
        unsigned char b = px[i * 4 + 2];
        unsigned char a = px[i * 4 + 3];
        out.argb_pixels[i] = (a << 24) | (r << 16) | (g << 8) | b;
    }
    return out;
}
}  // namespace

std::optional<rp::LoadedImage> TextureManager::load_texture_data(const char* filename) {
    int w, h, c;
    unsigned char* px = stbi_load(filename, &w, &h, &c, 4);
    if (!px) return std::nullopt;
    rp::LoadedImage img = pack_argb(px, w, h);
    stbi_image_free(px);
    return img;
}

std::optional<rp::LoadedImage> TextureManager::load_texture_data(std::span<const uint8_t> bytes) {
    int w, h, c;
    unsigned char* px = stbi_load_from_memory(
        bytes.data(), int(bytes.size()), &w, &h, &c, 4);
    if (!px) return std::nullopt;
    rp::LoadedImage img = pack_argb(px, w, h);
    stbi_image_free(px);
    return img;
}

}  // namespace plce::vk
