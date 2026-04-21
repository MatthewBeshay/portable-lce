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
    // flags packs:
    //   [0]     textured
    //   [1]     alpha_test
    //   [2]     lm_active
    //   [4:15]  tex_id    (12 bits — sampled image slot for diffuse)
    //   [16:27] lm_tex_id (12 bits — sampled image slot for lightmap)
    uint flags;
    uint global_lm_packed;
} pc;

layout(location = 0) out vec4 out_color;

void main() {
    uint tex_id    = (pc.flags >> 4u)  & 0xFFFu;
    uint lm_tex_id = (pc.flags >> 16u) & 0xFFFu;

    vec4 tex;
    if ((pc.flags & 1u) != 0u) {
        if ((pc.flags & 8u) != 0u) {
            // Forced LOD — StateSetForceLOD maps flags[28:31] → mipmap
            // level. Used by ItemInHandRenderer / ItemRenderer for
            // fixed-mip item textures (32×32 / 64×64 swap).
            float lod = float((pc.flags >> 28u) & 0xFu);
            tex = textureLod(sampler2D(u_images[nonuniformEXT(tex_id)], u_diffuse_sampler), v_uv, lod);
        } else {
            tex = texture(sampler2D(u_images[nonuniformEXT(tex_id)], u_diffuse_sampler), v_uv);
        }
    } else {
        tex = vec4(1.0);
    }
    vec4 c = tex * v_color;

    if ((pc.flags & 2u) != 0u && c.a < pc.alpha_ref)
        discard;

    if ((pc.flags & 4u) != 0u)
        c.rgb *= texture(sampler2D(u_images[nonuniformEXT(lm_tex_id)], u_lightmap_sampler), v_uv1).rgb;

    if (pc.fog_params.x > 0.5)
        c.rgb = mix(pc.fog_colour.rgb, c.rgb, v_fog_factor);

    c.rgb = pow(c.rgb, vec3(pc.inv_gamma));

    out_color = c;
}
