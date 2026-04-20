#pragma once

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include <glm/glm.hpp>

#include "platform/renderer/IRenderPath.h"

namespace plce::vk {

// Push constant block -- matches the GLSL layout in basic.vert / basic.frag.
// Single 256-byte range shared by both vertex and fragment stages.
struct alignas(16) PushConstants {
    glm::mat4 mvp;              // 0
    glm::vec4 nm0;              // 64   mat3(mv)[0].xyz, tex_scale_x
    glm::vec4 nm1;              // 80   mat3(mv)[1].xyz, tex_scale_y
    glm::vec4 nm2;              // 96   mat3(mv)[2].xyz, tex_offset_x
    glm::vec4 chunk_lit;        // 112  chunk_offset.xyz, lighting_enabled
    glm::vec4 l0;               // 128  light0_dir.xyz, tex_offset_y
    glm::vec4 l1;               // 144  light1_dir.xyz, mv_translation.x
    glm::vec4 ldiff;            // 160  light_diffuse.xyz, mv_translation.y
    glm::vec4 lamb;             // 176  light_ambient.xyz, mv_translation.z
    glm::vec4 fog_params;       // 192  mode, start, end, density
    glm::vec4 state_colour;     // 208
    glm::vec4 fog_colour;       // 224
    float     alpha_ref;        // 240
    float     inv_gamma;        // 244
    uint32_t  flags;            // 248  bit0=textured, bit1=alpha_test, bit2=lightmap
    uint32_t  global_lm_packed; // 252
};
static_assert(sizeof(PushConstants) == 256);
static_assert(offsetof(PushConstants, mvp) == 0);
static_assert(offsetof(PushConstants, fog_params) == 192);
static_assert(offsetof(PushConstants, state_colour) == 208);
static_assert(offsetof(PushConstants, fog_colour) == 224);
static_assert(offsetof(PushConstants, alpha_ref) == 240);
static_assert(offsetof(PushConstants, inv_gamma) == 244);
static_assert(offsetof(PushConstants, flags) == 248);
static_assert(offsetof(PushConstants, global_lm_packed) == 252);

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

// Convert triangle fan to triangle list on CPU.
inline std::vector<std::byte> fan_to_list(const void* data, int& count) {
    constexpr uint32_t kStride = 32;
    if (count < 3) { count = 0; return {}; }
    int tri_count = count - 2;
    int tri_verts = tri_count * 3;
    std::vector<std::byte> out(size_t(tri_verts) * kStride);
    const std::byte* src = static_cast<const std::byte*>(data);
    for (int i = 0; i < tri_count; ++i) {
        std::memcpy(out.data() + (i * 3 + 0) * kStride, src, kStride);
        std::memcpy(out.data() + (i * 3 + 1) * kStride, src + (i + 1) * kStride, kStride);
        std::memcpy(out.data() + (i * 3 + 2) * kStride, src + (i + 2) * kStride, kStride);
    }
    count = tri_verts;
    return out;
}

}  // namespace plce::vk
