#version 450
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_nonuniform_qualifier : require

#include "common.glsl"

layout(location = 0) in vec2  v_uv;
layout(location = 1) in vec4  v_color;
layout(location = 2) in float v_fog_factor;
layout(location = 3) in vec2  v_uv1;       // lightmap UV

// Bindless descriptor set (set = 0):
//   binding 0: sampled image array (per texture slot)
//   binding 1: immutable sampler[4] — (nearest|linear) × (repeat|clamp).
//              Index 3 (linear+clamp, single-mip) doubles as the lightmap
//              sampler; diffuse samples pick an index from PC::flags[30:31].
layout(set = 0, binding = 0) uniform texture2D u_images[];
layout(set = 0, binding = 1) uniform sampler   u_samplers[4];

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
