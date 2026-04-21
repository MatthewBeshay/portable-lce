#pragma once

// VertexFormats.h now aggregates two smaller headers:
//   - PushConstants.h        — the PushConstants struct + layout asserts
//     (small, stable, widely included).
//   - Narrow helpers below   — fan_to_list, blend_to_vk, depth_to_vk
//     (pull in Vulkan + IRenderPath; narrow use).
//
// Callers that only need the push-constant layout should include
// PushConstants.h directly. This aggregate header stays for source-
// compatibility with the two TUs that use everything
// (DisplayListManager.cpp, Renderer.cpp).

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

#include "PushConstants.h"
#include "platform/renderer/IRenderPath.h"

namespace plce::vk {

inline uint8_t blend_to_vk(rp::BlendFactor f) {
    using BF = rp::BlendFactor;
    switch (f) {
        case BF::zero:                     return VK_BLEND_FACTOR_ZERO;
        case BF::one:                      return VK_BLEND_FACTOR_ONE;
        case BF::src_color:                return VK_BLEND_FACTOR_SRC_COLOR;
        case BF::one_minus_src_color:      return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
        case BF::src_alpha:                return VK_BLEND_FACTOR_SRC_ALPHA;
        case BF::one_minus_src_alpha:      return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        case BF::dst_color:                return VK_BLEND_FACTOR_DST_COLOR;
        case BF::one_minus_dst_color:      return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
        case BF::dst_alpha:                return VK_BLEND_FACTOR_DST_ALPHA;
        case BF::one_minus_dst_alpha:      return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
        case BF::constant_alpha:           return VK_BLEND_FACTOR_CONSTANT_ALPHA;
        case BF::one_minus_constant_alpha: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
    }
    return VK_BLEND_FACTOR_ONE;
}

inline uint8_t depth_to_vk(rp::DepthTest f) {
    using DT = rp::DepthTest;
    switch (f) {
        case DT::off:           return VK_COMPARE_OP_ALWAYS;
        case DT::less:          return VK_COMPARE_OP_LESS;
        case DT::less_equal:    return VK_COMPARE_OP_LESS_OR_EQUAL;
        case DT::equal:         return VK_COMPARE_OP_EQUAL;
        case DT::greater:       return VK_COMPARE_OP_GREATER;
        case DT::greater_equal: return VK_COMPARE_OP_GREATER_OR_EQUAL;
        case DT::always:        return VK_COMPARE_OP_ALWAYS;
    }
    return VK_COMPARE_OP_LESS_OR_EQUAL;
}

struct FanToListResult {
    std::span<const std::byte> data;      // bytes (vertex_count * 32); empty if degenerate
    int                        vertex_count = 0;
};

// Convert a triangle fan (32-byte vertices) to a triangle list on CPU.
// Returns a view into a thread-local scratch buffer — valid only until
// the next fan_to_list call on the same thread. Avoids the per-call
// std::vector<std::byte> heap alloc the old API paid.
inline FanToListResult fan_to_list(const void* data, int in_count) {
    constexpr uint32_t kStride = 32;
    if (in_count < 3) return {};
    const int tri_count = in_count - 2;
    const int tri_verts = tri_count * 3;

    thread_local std::vector<std::byte> scratch;
    scratch.resize(size_t(tri_verts) * kStride);

    const std::byte* src = static_cast<const std::byte*>(data);
    for (int i = 0; i < tri_count; ++i) {
        std::memcpy(scratch.data() + (i * 3 + 0) * kStride, src, kStride);
        std::memcpy(scratch.data() + (i * 3 + 1) * kStride, src + (i + 1) * kStride, kStride);
        std::memcpy(scratch.data() + (i * 3 + 2) * kStride, src + (i + 2) * kStride, kStride);
    }
    return FanToListResult{std::span<const std::byte>(scratch.data(), scratch.size()), tri_verts};
}

}  // namespace plce::vk
