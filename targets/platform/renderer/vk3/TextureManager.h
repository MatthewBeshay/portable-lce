#pragma once

#include <vulkan/vulkan.h>
#include <vma/vk_mem_alloc.h>

#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "DeletionQueue.h"

namespace plce::vk3 {

// Per-texture sampler state
struct SamplerKey {
    VkFilter     min_filter = VK_FILTER_NEAREST;
    VkFilter     mag_filter = VK_FILTER_NEAREST;
    VkSamplerMipmapMode mip_mode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    VkSamplerAddressMode wrap_s = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    VkSamplerAddressMode wrap_t = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    bool operator==(const SamplerKey&) const = default;
};

struct SamplerKeyHash {
    size_t operator()(const SamplerKey& k) const noexcept {
        size_t h = 0;
        h ^= std::hash<int>{}(int(k.min_filter));
        h ^= std::hash<int>{}(int(k.mag_filter)) << 4;
        h ^= std::hash<int>{}(int(k.mip_mode)) << 8;
        h ^= std::hash<int>{}(int(k.wrap_s)) << 12;
        h ^= std::hash<int>{}(int(k.wrap_t)) << 16;
        return h;
    }
};

struct TextureSlot {
    VkImage         image    = VK_NULL_HANDLE;
    VmaAllocation   alloc    = nullptr;
    VkImageView     view     = VK_NULL_HANDLE;
    VkDescriptorSet desc_set = VK_NULL_HANDLE;
    uint32_t        width = 0, height = 0;
    bool            ready = false;
    SamplerKey      sampler_key{};
    bool            sampler_dirty = false;
    int             bound_lm = -1;  // lightmap index last written to desc binding 1
};

class TextureManager {
public:
    struct BoundTexResult { VkDescriptorSet ds; bool textured; bool lm_active; };

    void init(VkDevice device, VmaAllocator allocator, VkQueue queue, uint32_t queue_family,
              VkDescriptorSetLayout tex_set_layout, VkDescriptorPool tex_pool);
    void destroy(VkDevice device, VmaAllocator allocator);

    int create();
    void free(int idx, DeletionQueue& deletions);
    void bind(int idx);
    void bind_vertex(int idx) { lightmap_tex_ = idx; }
    void data(int w, int h, const void* pixels, int level);
    void data_update(int xo, int yo, int w, int h, const void* data, int level);
    void set_param(int param, int value);

    BoundTexResult bind_textures(VkCommandBuffer cmd, VkPipelineLayout layout);

    /// Poll pending upload fences, mark completed textures ready, free staging.
    /// Called once per frame from Renderer::StartFrame.
    void poll_uploads();

    int load_texture_data(const char* fn, void* info, int** out);
    int load_texture_data(uint8_t* data, uint32_t bytes, void* info, int** out);

    int default_tex() const { return default_tex_; }
    int default_lm() const { return default_lm_; }
    int bound_tex() const { return bound_tex_; }
    int lightmap_tex() const { return lightmap_tex_; }

private:
    /// Tracks an in-flight texture upload until its fence signals.
    struct PendingUpload {
        VkFence         fence         = VK_NULL_HANDLE;
        VkCommandBuffer cmd           = VK_NULL_HANDLE;
        VkBuffer        staging_buf   = VK_NULL_HANDLE;
        VmaAllocation   staging_alloc = nullptr;
        int             texture_idx   = -1;
    };

    VkSampler get_or_create_sampler(const SamplerKey& key);
    void update_tex_descriptor(TextureSlot& t);
    void upload_texture(int idx, int w, int h, const void* pixels);
    int ensure_default_texture();
    int ensure_default_lightmap();

    VkFence acquire_fence();
    void    release_fence(VkFence);
    void    complete_upload(PendingUpload& pu);
    void    wait_for_upload(int texture_idx);
    void    wait_all_uploads();

    VkDevice      device_       = VK_NULL_HANDLE;
    VmaAllocator  allocator_    = nullptr;
    VkQueue       queue_        = VK_NULL_HANDLE;
    VkDescriptorSetLayout tex_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool      tex_pool_       = VK_NULL_HANDLE;

    VkCommandPool upload_pool_ = VK_NULL_HANDLE;
    static constexpr VkDeviceSize kMaxUploadBytes = 64ull * 1024 * 1024;
    static constexpr uint32_t kDescPoolMaxSets = 4096;

    // Async upload tracking
    std::vector<PendingUpload> pending_uploads_;
    std::vector<VkFence>       fence_pool_;  // recycled fences

    // Descriptor pool capacity tracking
    uint32_t desc_sets_allocated_ = 0;
    bool     desc_pool_warned_    = false;

    // Sampler cache
    std::unordered_map<SamplerKey, VkSampler, SamplerKeyHash> sampler_cache_;

    // Texture storage
    std::vector<TextureSlot> textures_;
    int default_tex_ = 0;
    int default_lm_  = 0;
    int bound_tex_   = 0;
    int lightmap_tex_ = 0;
    SamplerKey lm_sampler_key_{VK_FILTER_LINEAR, VK_FILTER_LINEAR,
                               VK_SAMPLER_MIPMAP_MODE_NEAREST,
                               VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                               VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE};
    uint32_t next_material_id_ = 0;
    mutable std::mutex texture_mutex_;
};

}  // namespace plce::vk3
