#ifndef PLCE_COMMON_GLSL
#define PLCE_COMMON_GLSL

// Per-frame UBO (set = 1, binding = 0). Keep in sync with the FrameUBO
// struct in targets/platform/renderer/vk/FrameUBO.h — the C++ side
// has offsetof static_asserts that trip on layout drift.
layout(set = 1, binding = 0, std140) uniform FrameUBO {
    vec4  light0_dir;        // eye-space dir + pad
    vec4  light1_dir;
    vec4  light_diffuse;     // rgb + pad
    vec4  light_ambient;
    vec4  fog_params;        // mode, start, end, density
    vec4  fog_colour;        // rgb + pad in .w
    uint  global_lm_packed;  // low16=u, high16=v
    float inv_gamma;         // fragment tonemap
    uint  _pad0;
    uint  _pad1;
} frame;

// Shared 176-byte push constant block. Keep in sync with PushConstants
// struct in targets/platform/renderer/vk/PushConstants.h.
layout(push_constant) uniform PC {
    mat4  mvp;               // 0    proj * modelview
    vec4  nm0;               // 64   mat3(mv) col0 + .w = tex_scale_x
    vec4  nm1;               // 80   mat3(mv) col1 + .w = tex_scale_y
    vec4  nm2;               // 96   mat3(mv) col2 + .w = tex_offset_x
    vec4  chunk_lit;         // 112  chunk_offset.xyz + .w = lighting_enabled
    vec4  tex_mv;            // 128  tex_offset_y + mv_translation.xyz
    vec4  state_colour;      // 144
    float alpha_ref;         // 160
    uint  flags;             // 164
} pc;

// Flag bit layout for PC::flags. Keep in sync with fill_push_constants
// in Renderer.cpp.
//   [0]     FLAG_TEXTURED       diffuse sample enabled
//   [1]     FLAG_ALPHA_TEST     discard when alpha below alpha_ref
//   [2]     FLAG_LIGHTMAP       multiply c.rgb by lightmap sample
//   [3]     FLAG_FORCE_LOD      use textureLod with fixed level
//   [4:15]  TEX_ID              12-bit diffuse bindless slot (0..4095)
//   [16:27] LM_TEX_ID           12-bit lightmap bindless slot
//   [28:29] FORCE_LOD           2-bit LOD level
//   [30:31] SAMPLER_IDX         diffuse sampler index (0..3)
#define FLAG_TEXTURED       (1u << 0)
#define FLAG_ALPHA_TEST     (1u << 1)
#define FLAG_LIGHTMAP       (1u << 2)
#define FLAG_FORCE_LOD      (1u << 3)
#define TEX_ID_SHIFT        4u
#define TEX_ID_MASK         0xFFFu
#define LM_TEX_ID_SHIFT     16u
#define LM_TEX_ID_MASK      0xFFFu
#define FORCE_LOD_SHIFT     28u
#define FORCE_LOD_MASK      0x3u
#define SAMPLER_IDX_SHIFT   30u
#define SAMPLER_IDX_MASK    0x3u
// Fixed sampler index for the lightmap (linear+clamp, single mip).
#define LIGHTMAP_SAMPLER_IDX 3u

#endif  // PLCE_COMMON_GLSL
