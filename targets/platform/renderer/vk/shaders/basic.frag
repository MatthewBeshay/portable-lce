#version 450
#extension GL_EXT_nonuniform_qualifier : require

layout(location = 0) in vec2  v_uv;
layout(location = 1) in vec4  v_color;
layout(location = 2) in float v_fog_factor;
layout(location = 3) in vec2  v_uv1;       // lightmap UV

// Bindless descriptor set:
//   binding 0: sampled image array (per texture slot)
//   binding 1: immutable sampler for diffuse (nearest, mip-linear, repeat)
//   binding 2: immutable sampler for lightmap (linear, nearest-mip, clamp)
layout(set = 0, binding = 0) uniform texture2D u_images[];
layout(set = 0, binding = 1) uniform sampler   u_diffuse_sampler;
layout(set = 0, binding = 2) uniform sampler   u_lightmap_sampler;

// Same push constant block as vertex shader (shared range).
layout(push_constant) uniform PC {
    mat4 mvp;
    vec4 nm0, nm1, nm2;
    vec4 chunk_lit;
    vec4 l0, l1, ldiff, lamb;
    vec4 fog_params;
    vec4 state_colour;
    vec4 fog_colour;
    float alpha_ref;
    float inv_gamma;
    // See FLAG_* / *_SHIFT / *_MASK macros below for packed layout.
    uint flags;
    uint global_lm_packed;
} pc;

// PushConstants::flags bit layout (keep in sync with VertexFormats.h).
#define FLAG_TEXTURED       (1u << 0)
#define FLAG_ALPHA_TEST     (1u << 1)
#define FLAG_LIGHTMAP       (1u << 2)
#define FLAG_FORCE_LOD      (1u << 3)
#define TEX_ID_SHIFT        4u
#define TEX_ID_MASK         0xFFFu
#define LM_TEX_ID_SHIFT     16u
#define LM_TEX_ID_MASK      0xFFFu
#define FORCE_LOD_SHIFT     28u
#define FORCE_LOD_MASK      0xFu

layout(location = 0) out vec4 out_color;

void main() {
    uint tex_id    = (pc.flags >> TEX_ID_SHIFT)    & TEX_ID_MASK;
    uint lm_tex_id = (pc.flags >> LM_TEX_ID_SHIFT) & LM_TEX_ID_MASK;

    vec4 tex;
    if ((pc.flags & FLAG_TEXTURED) != 0u) {
        if ((pc.flags & FLAG_FORCE_LOD) != 0u) {
            // Forced LOD — StateSetForceLOD packs the level into
            // flags[28:31]. Used by ItemInHandRenderer / ItemRenderer
            // for fixed-mip item textures (32×32 / 64×64 swap).
            float lod = float((pc.flags >> FORCE_LOD_SHIFT) & FORCE_LOD_MASK);
            tex = textureLod(sampler2D(u_images[nonuniformEXT(tex_id)], u_diffuse_sampler), v_uv, lod);
        } else {
            tex = texture(sampler2D(u_images[nonuniformEXT(tex_id)], u_diffuse_sampler), v_uv);
        }
    } else {
        tex = vec4(1.0);
    }
    vec4 c = tex * v_color;

    if ((pc.flags & FLAG_ALPHA_TEST) != 0u && c.a < pc.alpha_ref)
        discard;

    if ((pc.flags & FLAG_LIGHTMAP) != 0u)
        c.rgb *= texture(sampler2D(u_images[nonuniformEXT(lm_tex_id)], u_lightmap_sampler), v_uv1).rgb;

    if (pc.fog_params.x > 0.5)
        c.rgb = mix(pc.fog_colour.rgb, c.rgb, v_fog_factor);

    c.rgb = pow(c.rgb, vec3(pc.inv_gamma));

    out_color = c;
}
