#version 450
#extension GL_EXT_nonuniform_qualifier : require

layout(location = 0) in vec2  v_uv;
layout(location = 1) in vec4  v_color;
layout(location = 2) in float v_fog_factor;
layout(location = 3) in vec2  v_uv1;       // lightmap UV

// Bindless descriptor set (set = 0):
//   binding 0: sampled image array (per texture slot)
//   binding 1: immutable sampler[4] — one entry per
//              (nearest|linear) x (repeat|clamp) combination.
//              Index 3 (linear+clamp, single-mip) doubles as the lightmap
//              sampler; diffuse samples pick an index from PushConstants::
//              flags bits [30:31].
layout(set = 0, binding = 0) uniform texture2D u_images[];
layout(set = 0, binding = 1) uniform sampler   u_samplers[4];

// Per-frame UBO (set = 1, binding = 0). See basic.vert for layout notes.
layout(set = 1, binding = 0, std140) uniform FrameUBO {
    vec4  light0_dir;
    vec4  light1_dir;
    vec4  light_diffuse;
    vec4  light_ambient;
    vec4  fog_params;
    vec4  fog_colour;        // rgb + pad in .w
    uint  global_lm_packed;
    float inv_gamma;
    uint  _pad0;
    uint  _pad1;
} frame;

// Same push constant block as vertex shader (shared range).
layout(push_constant) uniform PC {
    mat4 mvp;
    vec4 nm0, nm1, nm2;
    vec4 chunk_lit;
    vec4 tex_mv;
    vec4 state_colour;
    float alpha_ref;
    uint flags;
} pc;

// PushConstants::flags bit layout (keep in sync with VertexFormats.h).
//   [0]     FLAG_TEXTURED       diffuse sample enabled
//   [1]     FLAG_ALPHA_TEST     discard when alpha below ref
//   [2]     FLAG_LIGHTMAP       multiply c.rgb by lightmap sample
//   [3]     FLAG_FORCE_LOD      use textureLod with fixed level
//   [4:15]  TEX_ID              12-bit diffuse bindless slot (0..4095)
//   [16:27] LM_TEX_ID           12-bit lightmap bindless slot
//   [28:29] FORCE_LOD           2-bit LOD level (0..2 used by callers)
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

layout(location = 0) out vec4 out_color;

void main() {
    uint tex_id      = (pc.flags >> TEX_ID_SHIFT)      & TEX_ID_MASK;
    uint lm_tex_id   = (pc.flags >> LM_TEX_ID_SHIFT)   & LM_TEX_ID_MASK;
    uint sampler_idx = (pc.flags >> SAMPLER_IDX_SHIFT) & SAMPLER_IDX_MASK;

    vec4 tex;
    if ((pc.flags & FLAG_TEXTURED) != 0u) {
        if ((pc.flags & FLAG_FORCE_LOD) != 0u) {
            // Forced LOD — StateSetForceLOD packs the level into
            // flags[28:29]. Used by ItemInHandRenderer / ItemRenderer
            // for fixed-mip item textures (32×32 / 64×64 swap).
            float lod = float((pc.flags >> FORCE_LOD_SHIFT) & FORCE_LOD_MASK);
            tex = textureLod(sampler2D(u_images[nonuniformEXT(tex_id)], u_samplers[sampler_idx]), v_uv, lod);
        } else {
            tex = texture(sampler2D(u_images[nonuniformEXT(tex_id)], u_samplers[sampler_idx]), v_uv);
        }
    } else {
        tex = vec4(1.0);
    }
    vec4 c = tex * v_color;

    if ((pc.flags & FLAG_ALPHA_TEST) != 0u && c.a < pc.alpha_ref)
        discard;

    if ((pc.flags & FLAG_LIGHTMAP) != 0u)
        c.rgb *= texture(sampler2D(u_images[nonuniformEXT(lm_tex_id)], u_samplers[LIGHTMAP_SAMPLER_IDX]), v_uv1).rgb;

    if (frame.fog_params.x > 0.5)
        c.rgb = mix(frame.fog_colour.rgb, c.rgb, v_fog_factor);

    c.rgb = pow(c.rgb, vec3(frame.inv_gamma));

    out_color = c;
}
