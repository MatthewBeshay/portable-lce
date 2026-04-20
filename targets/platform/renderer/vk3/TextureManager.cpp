#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "TextureManager.h"
#include "VkCheck.h"
#include "VertexFormats.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

#include "platform/PlatformTypes.h"

namespace plce::vk3 {

void TextureManager::init(VkDevice device, VmaAllocator allocator, VkQueue queue,
                           uint32_t queue_family, VkDescriptorSetLayout tex_set_layout,
                           VkDescriptorPool tex_pool) {
    device_         = device;
    allocator_      = allocator;
    queue_          = queue;
    tex_set_layout_ = tex_set_layout;
    tex_pool_       = tex_pool;

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
    // Drain any in-flight uploads before tearing down.
    wait_all_uploads();

    if (upload_pool_) vkDestroyCommandPool(device, upload_pool_, nullptr);
    for (VkFence f : fence_pool_) vkDestroyFence(device, f, nullptr);
    fence_pool_.clear();

    for (auto& t : textures_) {
        if (t.view)  vkDestroyImageView(device, t.view, nullptr);
        if (t.image) vmaDestroyImage(allocator, t.image, t.alloc);
    }
    for (auto& [k, s] : sampler_cache_)
        vkDestroySampler(device, s, nullptr);
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

void TextureManager::release_fence(VkFence f) {
    fence_pool_.push_back(f);
}

void TextureManager::complete_upload(PendingUpload& pu) {
    textures_[pu.texture_idx].ready = true;
    vmaDestroyBuffer(allocator_, pu.staging_buf, pu.staging_alloc);
    vkFreeCommandBuffers(device_, upload_pool_, 1, &pu.cmd);
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
// SAMPLER CACHE
// ===================================================================

VkSampler TextureManager::get_or_create_sampler(const SamplerKey& key) {
    auto it = sampler_cache_.find(key);
    if (it != sampler_cache_.end()) return it->second;

    VkSamplerCreateInfo ci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    ci.magFilter    = key.mag_filter;
    ci.minFilter    = key.min_filter;
    ci.mipmapMode   = key.mip_mode;
    ci.addressModeU = key.wrap_s;
    ci.addressModeV = key.wrap_t;
    ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    ci.maxLod       = VK_LOD_CLAMP_NONE;

    VkSampler sampler = VK_NULL_HANDLE;
    check(vkCreateSampler(device_, &ci, nullptr, &sampler), "sampler");
    sampler_cache_.emplace(key, sampler);
    return sampler;
}

// ===================================================================
// TEXTURE BINDING HELPER
// ===================================================================

TextureManager::BoundTexResult TextureManager::bind_textures(VkCommandBuffer cmd,
                                                              VkPipelineLayout layout) {
    VkDescriptorSet ds = VK_NULL_HANDLE;
    bool textured = false;
    bool lm_active = false;
    {
        std::lock_guard lk(texture_mutex_);

        auto resolve = [&](int idx) -> int {
            if (idx <= 0 || size_t(idx) >= textures_.size()) return default_tex_;
            auto& t = textures_[idx];
            if (!t.ready) {
                // Lazy fence wait: only block when the texture is actually needed.
                wait_for_upload(idx);
                if (!t.ready) return default_tex_;  // no pending upload, just unused
            }
            return idx;
        };

        int tex = resolve(bound_tex_);
        auto& ts = textures_[tex];
        if (ts.sampler_dirty) { update_tex_descriptor(ts); ts.sampler_dirty = false; }
        ds = ts.desc_set;
        textured = (tex != default_tex_);

        // Update lightmap binding only when the bound lightmap actually changes.
        // bound_lm is invalidated (-1) whenever update_tex_descriptor or
        // upload_texture overwrites the descriptor set's binding 1.
        if (ds && lightmap_tex_ > 0 && size_t(lightmap_tex_) < textures_.size()) {
            int lm_idx = resolve(lightmap_tex_);
            if (lm_idx != default_tex_) {
                if (ts.bound_lm != lm_idx) {
                    auto& lm = textures_[lm_idx];
                    VkSampler lm_sampler = get_or_create_sampler(lm_sampler_key_);
                    VkDescriptorImageInfo lm_dii{lm_sampler, lm.view,
                                                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
                    VkWriteDescriptorSet wd{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                    wd.dstSet = ds; wd.dstBinding = 1; wd.descriptorCount = 1;
                    wd.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                    wd.pImageInfo = &lm_dii;
                    vkUpdateDescriptorSets(device_, 1, &wd, 0, nullptr);
                    ts.bound_lm = lm_idx;
                }
                lm_active = true;
            }
        }
    }
    if (ds) vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                     layout, 0, 1, &ds, 0, nullptr);
    return {ds, textured, lm_active};
}

// ===================================================================
// TEXTURES
// ===================================================================

int TextureManager::create() {
    std::lock_guard lk(texture_mutex_);
    textures_.emplace_back();
    return int(textures_.size()) - 1;
}

void TextureManager::free(int idx, DeletionQueue& deletions) {
    std::lock_guard lk(texture_mutex_);
    if (idx <= 0 || size_t(idx) >= textures_.size()) return;
    if (idx == default_tex_) return;
    // If there's a pending upload for this texture, wait for it first so
    // the image is no longer referenced by the GPU.
    wait_for_upload(idx);
    auto& t = textures_[idx];
    if (t.view || t.image) {
        auto view = t.view; auto img = t.image; auto alloc = t.alloc;
        auto dev = device_; auto vma = allocator_;
        deletions.push([=]() {
            if (view) vkDestroyImageView(dev, view, nullptr);
            if (img)  vmaDestroyImage(vma, img, alloc);
        });
    }
    t = {};
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
    {
        std::lock_guard lk(texture_mutex_);
        idx = bound_tex_;
        if (idx <= 0 || size_t(idx) >= textures_.size()) return;
        // Partial update assumes the image is in SHADER_READ_ONLY_OPTIMAL.
        // Wait for any pending full upload first.
        wait_for_upload(idx);
    }
    auto& t = textures_[idx];
    if (!t.ready) return;

    if (xo == 0 && yo == 0 && uint32_t(w) == t.width && uint32_t(h) == t.height) {
        // Full rewrite — use async upload path.
        upload_texture(idx, w, h, data);
        return;
    }
    if (xo < 0 || yo < 0 || uint32_t(xo + w) > t.width || uint32_t(yo + h) > t.height) return;

    // Partial sub-region update — async with per-upload staging + fence.
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

    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = upload_pool_; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VkCommandBuffer cmd; check(vkAllocateCommandBuffers(device_, &cai, &cmd), "tex update cmd");
    VkCommandBufferBeginInfo bbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bbi);

    // Transition to TRANSFER_DST for the sub-region copy
    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    b.srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    b.srcAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    b.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    b.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b.image = t.image;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dep.imageMemoryBarrierCount = 1; dep.pImageMemoryBarriers = &b;
    vkCmdPipelineBarrier2(cmd, &dep);

    VkBufferImageCopy rgn{};
    rgn.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    rgn.imageOffset = {xo, yo, 0};
    rgn.imageExtent = {uint32_t(w), uint32_t(h), 1};
    vkCmdCopyBufferToImage(cmd, stg_buf, t.image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &rgn);

    // Transition back to SHADER_READ_ONLY
    b.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    b.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    b.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    b.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    vkCmdPipelineBarrier2(cmd, &dep);

    vkEndCommandBuffer(cmd);

    VkFence fence = acquire_fence();
    VkCommandBufferSubmitInfo csi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    csi.commandBuffer = cmd;
    VkSubmitInfo2 sub{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    sub.commandBufferInfoCount = 1; sub.pCommandBufferInfos = &csi;
    check(vkQueueSubmit2(queue_, 1, &sub, fence), "tex update submit");

    // Texture is still "ready" (it had prior content), but we must mark the
    // upload pending so subsequent data_update calls wait before reusing.
    std::lock_guard lk(texture_mutex_);
    pending_uploads_.push_back({fence, cmd, stg_buf, stg_alloc, idx});
    textures_[idx].ready = false;  // will be remarked ready by complete_upload
}

void TextureManager::set_param(int param, int value) {
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
                case GL_NEAREST:
                    t.sampler_key.min_filter = VK_FILTER_NEAREST;
                    t.sampler_key.mip_mode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
                    break;
                case GL_LINEAR:
                    t.sampler_key.min_filter = VK_FILTER_LINEAR;
                    t.sampler_key.mip_mode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
                    break;
                case GL_NEAREST_MIPMAP_NEAREST:
                    t.sampler_key.min_filter = VK_FILTER_NEAREST;
                    t.sampler_key.mip_mode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
                    break;
                case GL_LINEAR_MIPMAP_NEAREST:
                    t.sampler_key.min_filter = VK_FILTER_LINEAR;
                    t.sampler_key.mip_mode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
                    break;
                case GL_NEAREST_MIPMAP_LINEAR:
                    t.sampler_key.min_filter = VK_FILTER_NEAREST;
                    t.sampler_key.mip_mode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
                    break;
                case GL_LINEAR_MIPMAP_LINEAR:
                    t.sampler_key.min_filter = VK_FILTER_LINEAR;
                    t.sampler_key.mip_mode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
                    break;
            }
            t.sampler_dirty = true;
            break;

        case GL_TEXTURE_MAG_FILTER:
            t.sampler_key.mag_filter = (value == GL_NEAREST)
                ? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
            t.sampler_dirty = true;
            break;

        case GL_TEXTURE_WRAP_S:
            t.sampler_key.wrap_s = (value == GL_CLAMP_TO_EDGE)
                ? VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE
                : VK_SAMPLER_ADDRESS_MODE_REPEAT;
            t.sampler_dirty = true;
            break;

        case GL_TEXTURE_WRAP_T:
            t.sampler_key.wrap_t = (value == GL_CLAMP_TO_EDGE)
                ? VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE
                : VK_SAMPLER_ADDRESS_MODE_REPEAT;
            t.sampler_dirty = true;
            break;
    }
}

int TextureManager::ensure_default_texture() {
    int idx = create();
    uint32_t pixel = 0xFFFFFFFF;
    { std::lock_guard lk(texture_mutex_); bound_tex_ = idx; }
    upload_texture(idx, 1, 1, &pixel);
    // Init path must have the texture ready before any rendering.
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

void TextureManager::update_tex_descriptor(TextureSlot& t) {
    if (!t.desc_set || !t.view) return;
    VkSampler sampler = get_or_create_sampler(t.sampler_key);
    VkDescriptorImageInfo dii{sampler, t.view,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet writes[2]{};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = t.desc_set; writes[0].dstBinding = 0; writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[0].pImageInfo = &dii;
    uint32_t n = 1;
    // Also write binding 1 with default lightmap fallback
    VkDescriptorImageInfo lm_dii{};
    if (default_lm_ > 0 && size_t(default_lm_) < textures_.size() && textures_[default_lm_].view) {
        auto& lm = textures_[default_lm_];
        VkSampler lm_sampler = get_or_create_sampler(lm_sampler_key_);
        lm_dii = {lm_sampler, lm.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = t.desc_set; writes[1].dstBinding = 1; writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[1].pImageInfo = &lm_dii;
        n = 2;
    }
    vkUpdateDescriptorSets(device_, n, writes, 0, nullptr);
    // Binding 1 was overwritten with default lightmap — invalidate cache
    t.bound_lm = -1;
}

void TextureManager::upload_texture(int idx, int w, int h, const void* pixels) {
    // If a previous upload for this texture is still in flight, wait for it
    // so we can safely destroy the old image.
    {
        std::lock_guard lk(texture_mutex_);
        wait_for_upload(idx);
    }

    auto& t = textures_[idx];
    bool reuse = t.ready && t.width == uint32_t(w) && t.height == uint32_t(h);
    if (t.ready && !reuse) {
        // Safe to destroy synchronously: the prior upload's fence has been waited on.
        if (t.view) vkDestroyImageView(device_, t.view, nullptr);
        if (t.image) vmaDestroyImage(allocator_, t.image, t.alloc);
        VkDescriptorSet keep = t.desc_set;
        SamplerKey sk = t.sampler_key;
        t = {}; t.desc_set = keep; t.sampler_key = sk;
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

    if (!t.desc_set) {
        VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dai.descriptorPool = tex_pool_; dai.descriptorSetCount = 1;
        dai.pSetLayouts = &tex_set_layout_;
        check(vkAllocateDescriptorSets(device_, &dai, &t.desc_set), "texture desc set");
    }
    // Write descriptor with per-texture sampler (binding 0 = diffuse, binding 1 = lightmap fallback)
    {
        VkSampler sampler = get_or_create_sampler(t.sampler_key);
        VkDescriptorImageInfo dii{sampler, t.view,
                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkWriteDescriptorSet writes[2]{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = t.desc_set; writes[0].dstBinding = 0; writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[0].pImageInfo = &dii;
        uint32_t n = 1;
        VkDescriptorImageInfo lm_dii{};
        if (default_lm_ > 0 && size_t(default_lm_) < textures_.size() && textures_[default_lm_].view) {
            auto& lm = textures_[default_lm_];
            VkSampler lm_sampler = get_or_create_sampler(lm_sampler_key_);
            lm_dii = {lm_sampler, lm.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[1].dstSet = t.desc_set; writes[1].dstBinding = 1; writes[1].descriptorCount = 1;
            writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[1].pImageInfo = &lm_dii;
            n = 2;
        }
        vkUpdateDescriptorSets(device_, n, writes, 0, nullptr);
    }
    t.sampler_dirty = false;
    t.bound_lm = -1;  // binding 1 was overwritten with default lightmap

    // Per-upload staging allocation (async-safe, no shared buffer contention)
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

    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = upload_pool_; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VkCommandBuffer cmd; check(vkAllocateCommandBuffers(device_, &cai, &cmd), "texture upload cmd");
    VkCommandBufferBeginInfo bbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bbi);

    // UNDEFINED -> TRANSFER_DST
    {
        VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        b.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
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

    VkFence fence = acquire_fence();
    VkCommandBufferSubmitInfo csi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    csi.commandBuffer = cmd;
    VkSubmitInfo2 sub{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    sub.commandBufferInfoCount = 1; sub.pCommandBufferInfos = &csi;
    check(vkQueueSubmit2(queue_, 1, &sub, fence), "texture upload submit");

    // Texture becomes ready only when the fence signals (poll_uploads or lazy wait).
    std::lock_guard lk(texture_mutex_);
    pending_uploads_.push_back({fence, cmd, stg_buf, stg_alloc, idx});
}

int TextureManager::load_texture_data(const char* fn, void* srcInfo, int** out) {
    int w, h, c;
    unsigned char* px = stbi_load(fn, &w, &h, &c, 4);
    if (!px) return -1;
    if (auto* i = static_cast<D3DXIMAGE_INFO*>(srcInfo)) { i->Width = w; i->Height = h; }
    *out = stb_to_argb(px, w, h);
    stbi_image_free(px);
    return 0;
}

int TextureManager::load_texture_data(uint8_t* data, uint32_t bytes, void* srcInfo, int** out) {
    int w, h, c;
    unsigned char* px = stbi_load_from_memory(data, int(bytes), &w, &h, &c, 4);
    if (!px) return -1;
    if (auto* i = static_cast<D3DXIMAGE_INFO*>(srcInfo)) { i->Width = w; i->Height = h; }
    *out = stb_to_argb(px, w, h);
    stbi_image_free(px);
    return 0;
}

}  // namespace plce::vk3
