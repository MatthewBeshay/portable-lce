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

// Expand compact 16-byte vertex format to 32-byte world_standard.
inline std::vector<std::byte> expand_compact(const void* data, int& count) {
    constexpr uint32_t kStride = 32;
    int quads = count / 4;
    int tri_verts = quads * 6;
    std::vector<std::byte> out(size_t(tri_verts) * kStride);
    const int16_t* src = static_cast<const int16_t*>(data);
    for (int q = 0; q < quads; ++q) {
        std::byte expanded[4 * kStride];
        for (int v = 0; v < 4; ++v) {
            const int16_t* sv = src + q * 4 * 8 + v * 8;
            auto* dst = expanded + v * kStride;
            auto* dstF = reinterpret_cast<float*>(dst);
            dstF[0] = sv[0] / 1024.0f;
            dstF[1] = sv[1] / 1024.0f;
            dstF[2] = sv[2] / 1024.0f;
            dstF[3] = sv[4] / 8192.0f;
            dstF[4] = sv[5] / 8192.0f;
            uint16_t packed = uint16_t(int(sv[3]) + 32768);
            dst[20] = std::byte(uint8_t((packed & 0x1F) * 255 / 31));
            dst[21] = std::byte(uint8_t(((packed >> 5) & 0x3F) * 255 / 63));
            dst[22] = std::byte(uint8_t(((packed >> 11) & 0x1F) * 255 / 31));
            dst[23] = std::byte(255);
            dst[24] = std::byte(0); dst[25] = std::byte(127);
            dst[26] = std::byte(0); dst[27] = std::byte(0);
            auto* dstS = reinterpret_cast<int16_t*>(dst + 28);
            dstS[0] = sv[6]; dstS[1] = sv[7];
        }
        auto put = [&](int ti, int vi) {
            std::memcpy(out.data() + (q * 6 + ti) * kStride,
                        expanded + vi * kStride, kStride);
        };
        put(0,0); put(1,1); put(2,2); put(3,0); put(4,2); put(5,3);
    }
    count = tri_verts;
    return out;
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
