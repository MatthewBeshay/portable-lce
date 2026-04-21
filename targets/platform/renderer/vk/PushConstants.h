#pragma once

#include <cstddef>
#include <cstdint>

#include <glm/glm.hpp>

namespace plce::vk {

// Push constant block — matches the GLSL layout in basic.vert / basic.frag.
// 176-byte range shared by both vertex and fragment stages. Per-frame state
// (lights, fog, gamma, global lightmap) lives in the set=1 FrameUBO, not
// here — only per-draw data stays in push constants.
//
// `flags` is packed per the FLAG_* / *_SHIFT / *_MASK macros in basic.frag
// (FLAG_TEXTURED / FLAG_ALPHA_TEST / FLAG_LIGHTMAP / FLAG_FORCE_LOD /
// TEX_ID_SHIFT / LM_TEX_ID_SHIFT / FORCE_LOD_SHIFT / SAMPLER_IDX_SHIFT).
struct alignas(16) PushConstants {
    glm::mat4 mvp;              // 0
    glm::vec4 nm0;              // 64   mat3(mv)[0].xyz, tex_scale_x
    glm::vec4 nm1;              // 80   mat3(mv)[1].xyz, tex_scale_y
    glm::vec4 nm2;              // 96   mat3(mv)[2].xyz, tex_offset_x
    glm::vec4 chunk_lit;        // 112  chunk_offset.xyz, lighting_enabled
    glm::vec4 tex_mv;           // 128  tex_offset_y, mv_translation.xyz
    glm::vec4 state_colour;     // 144
    float     alpha_ref;        // 160
    uint32_t  flags;            // 164
    uint32_t  _pad0;            // 168
    uint32_t  _pad1;            // 172
};
static_assert(sizeof(PushConstants) == 176);
static_assert(offsetof(PushConstants, mvp)          == 0);
static_assert(offsetof(PushConstants, tex_mv)       == 128);
static_assert(offsetof(PushConstants, state_colour) == 144);
static_assert(offsetof(PushConstants, alpha_ref)    == 160);
static_assert(offsetof(PushConstants, flags)        == 164);

}  // namespace plce::vk
