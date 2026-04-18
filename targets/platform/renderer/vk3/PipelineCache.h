#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstring>
#include <unordered_map>

namespace plce::vk3 {

/// Pipeline state key — one pipeline per unique render state combination.
/// Dynamic state covers: viewport, scissor, topology, blend constants, depth bias.
struct PsoKey {
    uint8_t depth_test   : 1 = 1;
    uint8_t depth_write  : 1 = 1;
    uint8_t blend_enable : 1 = 0;
    uint8_t cull_back    : 1 = 0;
    uint8_t lines        : 1 = 0;
    uint8_t pad0         : 3 = 0;
    uint8_t depth_func   = 3;  // VK_COMPARE_OP_LESS_OR_EQUAL
    uint8_t blend_src    = 6;  // VK_BLEND_FACTOR_SRC_ALPHA
    uint8_t blend_dst    = 7;  // VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA
    uint8_t color_mask   = 0x0F;
    bool operator==(const PsoKey&) const = default;
};

struct PsoKeyHash {
    size_t operator()(const PsoKey& k) const noexcept {
        uint64_t v = 0;
        std::memcpy(&v, &k, sizeof(k));
        return std::hash<uint64_t>{}(v);
    }
};

/// Creates and caches VkPipeline objects keyed by PsoKey.
class PipelineCache {
public:
    struct Config {
        VkDevice         device       = VK_NULL_HANDLE;
        VkPipelineLayout layout       = VK_NULL_HANDLE;
        VkFormat         color_format = VK_FORMAT_UNDEFINED;
        VkFormat         depth_format = VK_FORMAT_UNDEFINED;
        const uint32_t*  vert_spv     = nullptr;
        size_t           vert_size    = 0;
        const uint32_t*  frag_spv     = nullptr;
        size_t           frag_size    = 0;
    };

    void init(const Config& cfg);
    void destroy();

    /// Get or create pipeline for the given state key.
    VkPipeline get(const PsoKey& key);

    /// Pre-create common pipeline variants to avoid first-draw hitches.
    void warm_up();

private:
    VkPipeline create(const PsoKey& key);

    Config cfg_{};
    VkShaderModule vert_mod_ = VK_NULL_HANDLE;
    VkShaderModule frag_mod_ = VK_NULL_HANDLE;
    std::unordered_map<PsoKey, VkPipeline, PsoKeyHash> cache_;
};

}  // namespace plce::vk3
