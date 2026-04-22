#pragma once

#include <cstdint>

#include <glm/glm.hpp>

namespace plce::vk {

// Per-frame uniform block (set = 1, binding = 0) — lives in a host-visible
// UBO that the renderer fills once per StartFrame. Holds state that does
// not change within a frame, pulling it off the push-constant hot path.
//
// std140 layout: every member aligns to 16 bytes. fog_colour.w doubles as
// inv_gamma so the final tonemap can read a single vec4. Keep in sync with
// the `FrameUBO` block in basic.vert / basic.frag.
struct alignas(16) FrameUBO {
    glm::vec4 light0_dir;       // 0    eye-space direction + pad
    glm::vec4 light1_dir;       // 16
    glm::vec4 light_diffuse;    // 32   rgb + pad
    glm::vec4 light_ambient;    // 48
    glm::vec4 fog_params;       // 64   mode, start, end, density
    glm::vec4 fog_colour;       // 80   rgb + inv_gamma in .w
    uint32_t  global_lm_packed; // 96   low16=u, high16=v
    uint32_t  _pad0;            // 100
    uint32_t  _pad1;            // 104
    uint32_t  _pad2;            // 108
};

// std140 offsets. Mismatch between this struct and the GLSL FrameUBO
// block silently corrupts shading; these asserts trip at compile time
// before that ever ships.
static_assert(sizeof(FrameUBO) == 112);
static_assert(offsetof(FrameUBO, light0_dir)       == 0);
static_assert(offsetof(FrameUBO, light1_dir)       == 16);
static_assert(offsetof(FrameUBO, light_diffuse)    == 32);
static_assert(offsetof(FrameUBO, light_ambient)    == 48);
static_assert(offsetof(FrameUBO, fog_params)       == 64);
static_assert(offsetof(FrameUBO, fog_colour)       == 80);
static_assert(offsetof(FrameUBO, global_lm_packed) == 96);

}  // namespace plce::vk
