#pragma once

#include <cstddef>
#include <cstdint>

#include <glm/glm.hpp>

namespace plce::vk {

// Push constant block — matches the GLSL layout in basic.vert / basic.frag.
// Single 256-byte range shared by both vertex and fragment stages.
//
// `flags` is packed per the FLAG_* / *_SHIFT / *_MASK macros in basic.frag
// (FLAG_TEXTURED / FLAG_ALPHA_TEST / FLAG_LIGHTMAP / FLAG_FORCE_LOD /
// TEX_ID_SHIFT / LM_TEX_ID_SHIFT / FORCE_LOD_SHIFT).
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
    uint32_t  flags;            // 248
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

}  // namespace plce::vk
