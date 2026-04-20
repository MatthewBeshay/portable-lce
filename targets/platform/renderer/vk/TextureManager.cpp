#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "TextureManager.h"
#include "VkCheck.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace plce::vk {

void TextureManager::init(VkDevice device, VmaAllocator allocator, VkQueue queue,
                          uint32_t queue_family, VkDescriptorSet bindless_set) {
    device_        = device;
    allocator_     = allocator;
    queue_         = queue;
    bindless_set_  = bindless_set;

    textures_.reserve(256);
    pending_uploads_.reserve(16);
    fence_pool_.reserve(16);

    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
                VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = queue_family;
    check(vkCreateCommandPool(device_, &pci, nullptr, &upload_pool_), "upload pool");

    default_tex_ = ensure_default_texture();
    default_lm_  = ensure_default_lightmap();
    bound_tex_   = default_tex_;
}

void TextureManager::destroy(VkDevice device, VmaAllocator allocator) {
    wait_all_uploads();

    if (upload_pool_) vkDestroyCommandPool(device, upload_pool_, nullptr);
    for (VkFence f : fence_pool_) vkDestroyFence(device, f, nullptr);
    fence_pool_.clear();

    for (auto& t : textures_) {
        if (t.view)  vkDestroyImageView(device, t.view, nullptr);
        if (t.image) vmaDestroyImage(allocator, t.image, t.alloc);
    }
}

// ===================================================================
// FENCE POOL
// ===================================================================

VkFence TextureManager::acquire_fence() {
    if (!fence_pool_.empty()) {
        VkFence f = fence_pool_.back();
        fence_pool_.pop_back();
        return f;
    }
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence f = VK_NULL_HANDLE;
    check(vkCreateFence(device_, &fci, nullptr, &f), "upload fence");
    return f;
}

void TextureManager::release_fence(VkFence f) { fence_pool_.push_back(f); }

void TextureManager::complete_upload(PendingUpload& pu) {
    textures_[pu.texture_idx].ready = true;
    // Publish the image view into the bindless array now that it's safe to sample.
    write_slot(pu.texture_idx);
    vmaDestroyBuffer(allocator_, pu.staging_buf, pu.staging_alloc);
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
    auto& t = textures_[idx];
    if (!t.view) return;

    VkDescriptorImageInfo dii{};
    dii.sampler     = VK_NULL_HANDLE;  // separate sampler binding
    dii.imageView   = t.view;
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
    std::lock_guard lk(texture_mutex_);
    int idx = bound_tex_;
    if (idx <= 0 || size_t(idx) >= textures_.size() || !textures_[idx].ready) {
        if (idx > 0 && size_t(idx) < textures_.size()) wait_for_upload(idx);
        if (idx <= 0 || size_t(idx) >= textures_.size() || !textures_[idx].ready) {
            textured_out = false;
            return uint32_t(default_tex_);
        }
    }
    textured_out = (idx != default_tex_);
    return uint32_t(idx);
}

uint32_t TextureManager::resolve_lightmap_slot(bool& active_out) {
    std::lock_guard lk(texture_mutex_);
    int idx = lightmap_tex_;
    if (idx <= 0 || size_t(idx) >= textures_.size() || !textures_[idx].ready) {
        if (idx > 0 && size_t(idx) < textures_.size()) wait_for_upload(idx);
        if (idx <= 0 || size_t(idx) >= textures_.size() || !textures_[idx].ready) {
            active_out = false;
            return uint32_t(default_lm_);
        }
    }
    active_out = true;
    return uint32_t(idx);
}

// ===================================================================
// TEXTURES (thin legacy API)
// ===================================================================

int TextureManager::create() {
    std::lock_guard lk(texture_mutex_);
    textures_.emplace_back();
    const int idx = int(textures_.size()) - 1;
    if (uint32_t(idx) >= kMaxTextures) {
        std::fprintf(stderr, "[vk] WARNING: texture count %d exceeds bindless array size %u\n",
                     idx, kMaxTextures);
    }
    return idx;
}

void TextureManager::free(int idx, DeletionQueue& deletions) {
    std::lock_guard lk(texture_mutex_);
    if (idx <= 0 || size_t(idx) >= textures_.size()) return;
    if (idx == default_tex_) return;
    wait_for_upload(idx);
    auto& t = textures_[idx];
    if (t.view || t.image)
        deletions.push_view_image(device_, t.view, allocator_, t.image, t.alloc);
    t = {};
    // Leave the bindless slot pointing at the stale view — PARTIALLY_BOUND
    // means it only matters if something samples it, which it shouldn't
    // after the free. Callers that reuse the slot via create() will
    // overwrite the entry on the next upload.
}

void TextureManager::bind(int idx) {
    std::lock_guard lk(texture_mutex_);
    if (idx < 0) { bound_tex_ = default_tex_; return; }
    if (size_t(idx) >= textures_.size()) textures_.resize(idx + 1);
    bound_tex_ = idx;
}

void TextureManager::data(int w, int h, const void* pixels, int level) {
    if (level != 0 || !pixels) return;
    int idx;
    { std::lock_guard lk(texture_mutex_); idx = bound_tex_; if (idx <= 0) return;
      if (size_t(idx) >= textures_.size()) textures_.resize(idx + 1); }

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

    VkBufferCreateInfo stg_bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    stg_bi.size = bytes; stg_bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VmaAllocationCreateInfo stg_ai{};
    stg_ai.usage = VMA_MEMORY_USAGE_AUTO;
    stg_ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                   VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VmaAllocationInfo stg_info{};
    VkBuffer stg_buf = VK_NULL_HANDLE; VmaAllocation stg_alloc = nullptr;
    check(vmaCreateBuffer(allocator_, &stg_bi, &stg_ai, &stg_buf, &stg_alloc, &stg_info),
          "tex update staging");
    std::memcpy(stg_info.pMappedData, data, bytes);

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
        rgn.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        rgn.imageOffset = {xo, yo, 0};
        rgn.imageExtent = {uint32_t(w), uint32_t(h), 1};
        vkCmdCopyBufferToImage(cmd, stg_buf, image,
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
    check(vkQueueSubmit2(queue_, 1, &sub, fence), "tex update submit");

    std::lock_guard lk(texture_mutex_);
    pending_uploads_.push_back({fence, cmd, stg_buf, stg_alloc, idx});
    textures_[idx].ready = false;
}

void TextureManager::set_param(int param, int value) {
    // Recorded on the slot for possible future per-texture sampler support,
    // but with the current single-sampler bindless layout this is a no-op
    // in terms of rendering.
    std::lock_guard lk(texture_mutex_);
    if (bound_tex_ <= 0 || size_t(bound_tex_) >= textures_.size()) return;
    auto& t = textures_[bound_tex_];

    constexpr int GL_TEXTURE_MIN_FILTER = 0x2801;
    constexpr int GL_TEXTURE_MAG_FILTER = 0x2800;
    constexpr int GL_TEXTURE_WRAP_S     = 0x2802;
    constexpr int GL_TEXTURE_WRAP_T     = 0x2803;
    constexpr int GL_NEAREST            = 0x2600;
    constexpr int GL_LINEAR             = 0x2601;
    constexpr int GL_NEAREST_MIPMAP_NEAREST = 0x2700;
    constexpr int GL_LINEAR_MIPMAP_NEAREST  = 0x2701;
    constexpr int GL_NEAREST_MIPMAP_LINEAR  = 0x2702;
    constexpr int GL_LINEAR_MIPMAP_LINEAR   = 0x2703;
    constexpr int GL_CLAMP_TO_EDGE = 0x812F;

    switch (param) {
        case GL_TEXTURE_MIN_FILTER:
            switch (value) {
                case GL_NEAREST:                 t.sampler_key.min_filter = VK_FILTER_NEAREST; t.sampler_key.mip_mode = VK_SAMPLER_MIPMAP_MODE_NEAREST; break;
                case GL_LINEAR:                  t.sampler_key.min_filter = VK_FILTER_LINEAR;  t.sampler_key.mip_mode = VK_SAMPLER_MIPMAP_MODE_NEAREST; break;
                case GL_NEAREST_MIPMAP_NEAREST:  t.sampler_key.min_filter = VK_FILTER_NEAREST; t.sampler_key.mip_mode = VK_SAMPLER_MIPMAP_MODE_NEAREST; break;
                case GL_LINEAR_MIPMAP_NEAREST:   t.sampler_key.min_filter = VK_FILTER_LINEAR;  t.sampler_key.mip_mode = VK_SAMPLER_MIPMAP_MODE_NEAREST; break;
                case GL_NEAREST_MIPMAP_LINEAR:   t.sampler_key.min_filter = VK_FILTER_NEAREST; t.sampler_key.mip_mode = VK_SAMPLER_MIPMAP_MODE_LINEAR;  break;
                case GL_LINEAR_MIPMAP_LINEAR:    t.sampler_key.min_filter = VK_FILTER_LINEAR;  t.sampler_key.mip_mode = VK_SAMPLER_MIPMAP_MODE_LINEAR;  break;
            }
            break;
        case GL_TEXTURE_MAG_FILTER:
            t.sampler_key.mag_filter = (value == GL_NEAREST) ? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
            break;
        case GL_TEXTURE_WRAP_S:
        case GL_TEXTURE_WRAP_T: {
            VkSamplerAddressMode mode;
            if (value == GL_CLAMP_TO_EDGE)        mode = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            else if (value == 0x2901 /*GL_REPEAT*/) mode = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            else {
                std::fprintf(stderr, "[vk] TextureSetParam: unsupported WRAP value 0x%x, using REPEAT\n", unsigned(value));
                mode = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            }
            if (param == GL_TEXTURE_WRAP_S) t.sampler_key.wrap_s = mode;
            else                            t.sampler_key.wrap_t = mode;
            break;
        }
    }
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
    {
        std::lock_guard lk(texture_mutex_);
        wait_for_upload(idx);
    }

    auto& t = textures_[idx];
    bool reuse = t.ready && t.width == uint32_t(w) && t.height == uint32_t(h);
    if (t.ready && !reuse) {
        if (t.view)  vkDestroyImageView(device_, t.view, nullptr);
        if (t.image) vmaDestroyImage(allocator_, t.image, t.alloc);
        SamplerKey sk = t.sampler_key;
        t = {}; t.sampler_key = sk;
    }
    t.width = w; t.height = h;

    uint32_t mips = 1;
    { uint32_t d = std::max(uint32_t(w), uint32_t(h)); while (d > 1) { d >>= 1; ++mips; } }

    if (!reuse) {
        VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format    = VK_FORMAT_R8G8B8A8_UNORM;
        ici.extent    = {t.width, t.height, 1};
        ici.mipLevels = mips; ici.arrayLayers = 1;
        ici.samples   = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling    = VK_IMAGE_TILING_OPTIMAL;
        ici.usage     = VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                        VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                        VK_IMAGE_USAGE_SAMPLED_BIT;
        VmaAllocationCreateInfo ai{}; ai.usage = VMA_MEMORY_USAGE_AUTO;
        check(vmaCreateImage(allocator_, &ici, &ai, &t.image, &t.alloc, nullptr), "texture image");

        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image = t.image; vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = VK_FORMAT_R8G8B8A8_UNORM;
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mips, 0, 1};
        check(vkCreateImageView(device_, &vci, nullptr, &t.view), "texture view");
    }

    VkDeviceSize bytes = VkDeviceSize(w) * h * 4;
    if (bytes > kMaxUploadBytes) return;

    VkBufferCreateInfo stg_bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    stg_bi.size = bytes; stg_bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VmaAllocationCreateInfo stg_ai{};
    stg_ai.usage = VMA_MEMORY_USAGE_AUTO;
    stg_ai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                   VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VmaAllocationInfo stg_info{};
    VkBuffer stg_buf = VK_NULL_HANDLE; VmaAllocation stg_alloc = nullptr;
    check(vmaCreateBuffer(allocator_, &stg_bi, &stg_ai, &stg_buf, &stg_alloc, &stg_info),
          "texture staging");
    std::memcpy(stg_info.pMappedData, pixels, bytes);

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
            b.image = t.image;
            b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mips, 0, 1};
            VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            dep.imageMemoryBarrierCount = 1; dep.pImageMemoryBarriers = &b;
            vkCmdPipelineBarrier2(cmd, &dep);
        }
        VkBufferImageCopy rgn{};
        rgn.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        rgn.imageExtent = {t.width, t.height, 1};
        vkCmdCopyBufferToImage(cmd, stg_buf, t.image,
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
            b.image = t.image;
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
            vkCmdBlitImage(cmd, t.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           t.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
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
                ends[n].image = t.image;
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
            ends[n].image = t.image;
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
    check(vkQueueSubmit2(queue_, 1, &sub, fence), "texture upload submit");

    std::lock_guard lk(texture_mutex_);
    pending_uploads_.push_back({fence, cmd, stg_buf, stg_alloc, idx});
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
