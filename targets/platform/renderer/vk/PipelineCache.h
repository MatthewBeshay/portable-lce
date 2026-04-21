#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

namespace plce::vk {

/// Pipeline state key — one pipeline per unique render state combination.
/// Packed into a single uint64_t with explicit bit offsets so the hash is
/// portable across compilers (bitfield layout is implementation-defined).
/// Dynamic state covers: viewport, scissor, topology, blend constants, depth bias.
struct PipelineKey {
    uint64_t bits = 0;

    // Bit layout:
    //   [0]     depth_test
    //   [1]     depth_write
    //   [2]     blend_enable
    //   [3]     cull_back
    //   [4]     stencil_test
    //   [5]     compact  — 16-byte packed vertex format (decoded in shader)
    //   [6:7]   reserved for future state bits (alpha-to-coverage, wireframe)
    //   [8:15]  depth_func    (VkCompareOp; 8 bits but values fit in 4)
    //   [16:23] blend_src    (VkBlendFactor)
    //   [24:31] blend_dst    (VkBlendFactor)
    //   [32:35] color_mask   (4 bits: R,G,B,A)
    //   [36:43] stencil_func (VkCompareOp)
    //   [44:63] reserved for future shader-permutation / stencil-op bits
    //
    // Line / triangle topology is a dynamic state
    // (VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY) — it does not affect the pipeline
    // object, so it has no bit here. Stencil compare/write mask and reference
    // are also dynamic.
    enum : uint64_t {
        kDepthTestShift   = 0,  kDepthTestMask   = 1ull << 0,
        kDepthWriteShift  = 1,  kDepthWriteMask  = 1ull << 1,
        kBlendEnableShift = 2,  kBlendEnableMask = 1ull << 2,
        kCullBackShift    = 3,  kCullBackMask    = 1ull << 3,
        kStencilTestShift = 4,  kStencilTestMask = 1ull << 4,
        kCompactShift     = 5,  kCompactMask     = 1ull << 5,
        kDepthFuncShift   = 8,  kDepthFuncMask   = 0xFFull << 8,
        kBlendSrcShift    = 16, kBlendSrcMask    = 0xFFull << 16,
        kBlendDstShift    = 24, kBlendDstMask    = 0xFFull << 24,
        kColorMaskShift   = 32, kColorMaskMask   = 0xFull << 32,
        kStencilFuncShift = 36, kStencilFuncMask = 0xFFull << 36,
    };

    constexpr bool depth_test()   const { return bits & kDepthTestMask; }
    constexpr bool depth_write()  const { return bits & kDepthWriteMask; }
    constexpr bool blend_enable() const { return bits & kBlendEnableMask; }
    constexpr bool cull_back()    const { return bits & kCullBackMask; }
    constexpr bool stencil_test() const { return bits & kStencilTestMask; }
    constexpr bool compact()      const { return bits & kCompactMask; }
    constexpr uint8_t depth_func()   const { return uint8_t((bits & kDepthFuncMask)   >> kDepthFuncShift); }
    constexpr uint8_t blend_src()    const { return uint8_t((bits & kBlendSrcMask)    >> kBlendSrcShift); }
    constexpr uint8_t blend_dst()    const { return uint8_t((bits & kBlendDstMask)    >> kBlendDstShift); }
    constexpr uint8_t color_mask()   const { return uint8_t((bits & kColorMaskMask)   >> kColorMaskShift); }
    constexpr uint8_t stencil_func() const { return uint8_t((bits & kStencilFuncMask) >> kStencilFuncShift); }

    void set_depth_test(bool v)   { bits = (bits & ~kDepthTestMask)   | (uint64_t(v) << kDepthTestShift); }
    void set_depth_write(bool v)  { bits = (bits & ~kDepthWriteMask)  | (uint64_t(v) << kDepthWriteShift); }
    void set_blend_enable(bool v) { bits = (bits & ~kBlendEnableMask) | (uint64_t(v) << kBlendEnableShift); }
    void set_cull_back(bool v)    { bits = (bits & ~kCullBackMask)    | (uint64_t(v) << kCullBackShift); }
    void set_stencil_test(bool v) { bits = (bits & ~kStencilTestMask) | (uint64_t(v) << kStencilTestShift); }
    void set_compact(bool v)      { bits = (bits & ~kCompactMask)     | (uint64_t(v) << kCompactShift); }
    void set_depth_func(uint8_t v)  { bits = (bits & ~kDepthFuncMask)   | (uint64_t(v) << kDepthFuncShift); }
    void set_blend_src(uint8_t v)   { bits = (bits & ~kBlendSrcMask)    | (uint64_t(v) << kBlendSrcShift); }
    void set_blend_dst(uint8_t v)   { bits = (bits & ~kBlendDstMask)    | (uint64_t(v) << kBlendDstShift); }
    void set_color_mask(uint8_t v)  { bits = (bits & ~kColorMaskMask)   | (uint64_t(v & 0xF) << kColorMaskShift); }
    void set_stencil_func(uint8_t v){ bits = (bits & ~kStencilFuncMask) | (uint64_t(v) << kStencilFuncShift); }

    constexpr PipelineKey() {
        // Default: depth test+write on, CMP=LESS_OR_EQUAL(3), blend=SRC_ALPHA/ONE_MINUS_SRC_ALPHA, color mask RGBA
        bits = kDepthTestMask | kDepthWriteMask
             | (uint64_t(3) << kDepthFuncShift)
             | (uint64_t(6) << kBlendSrcShift)
             | (uint64_t(7) << kBlendDstShift)
             | (uint64_t(0xF) << kColorMaskShift);
    }

    bool operator==(const PipelineKey&) const = default;
};

/// Creates and caches VkPipeline objects keyed by PipelineKey.
class PipelineCache {
public:
    struct Config {
        VkDevice         device                  = VK_NULL_HANDLE;
        VkPipelineLayout layout                  = VK_NULL_HANDLE;
        VkFormat         color_format            = VK_FORMAT_UNDEFINED;
        VkFormat         depth_format            = VK_FORMAT_UNDEFINED;
        VkFormat         stencil_format          = VK_FORMAT_UNDEFINED;
        const uint32_t*  vert_spv                = nullptr;
        size_t           vert_size               = 0;
        const uint32_t*  frag_spv                = nullptr;
        size_t           frag_size               = 0;
        const uint32_t*  vert_compact_spv        = nullptr;
        size_t           vert_compact_size       = 0;
    };

    void init(const Config& cfg);
    void destroy();

    /// Get or create pipeline for the given state key.
    VkPipeline get(const PipelineKey& key);

    /// Pre-create common pipeline variants to avoid first-draw hitches.
    void warm_up();

    /// Populate the VkPipelineCache from a previously serialized blob.
    /// If the blob is invalid or empty, falls back to an empty cache.
    void load_cache(std::span<const std::uint8_t> blob);

    /// Serialize the VkPipelineCache into a byte buffer the caller can
    /// persist wherever makes sense (a filesystem, the network, /dev/null).
    [[nodiscard]] std::vector<std::uint8_t> save_cache() const;

private:
    VkPipeline create(const PipelineKey& key);

    // Open-addressing table with Fibonacci hashing and linear probing.
    // Capacity is always a power of two; load factor is capped at 0.5 so
    // probe chains stay short. Empty slots have pipeline == VK_NULL_HANDLE
    // (pipeline handles created by vkCreateGraphicsPipelines are never
    // null on success, so VK_NULL_HANDLE is a safe sentinel).
    struct Entry {
        uint64_t   key      = 0;
        VkPipeline pipeline = VK_NULL_HANDLE;
    };
    void rehash(size_t new_cap);
    static size_t probe(const std::vector<Entry>& slots, size_t mask, uint64_t key);

    Config cfg_{};
    VkShaderModule vert_mod_         = VK_NULL_HANDLE;
    VkShaderModule vert_compact_mod_ = VK_NULL_HANDLE;
    VkShaderModule frag_mod_         = VK_NULL_HANDLE;
    VkPipelineCache vk_cache_ = VK_NULL_HANDLE;
    std::vector<Entry> cache_slots_;
    size_t cache_mask_ = 0;    // cache_slots_.size() - 1
    size_t cache_count_ = 0;
};

}  // namespace plce::vk
