#version 450

layout(location = 0) in vec2  v_uv;
layout(location = 1) in vec4  v_color;
layout(location = 2) in float v_fog_factor;
layout(location = 3) in vec2  v_uv1;       // lightmap UV

layout(set = 0, binding = 0) uniform sampler2D u_tex;
layout(set = 0, binding = 1) uniform sampler2D u_lm;   // lightmap (1x1 white fallback when unused)

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
    uint flags;             // bit0=textured, bit1=alpha_test, bit2=lightmap
    uint global_lm_packed;
} pc;

layout(location = 0) out vec4 out_color;

void main() {
    vec4 tex = ((pc.flags & 1u) != 0u) ? texture(u_tex, v_uv) : vec4(1.0);
    vec4 c = tex * v_color;

    // Alpha test
    if ((pc.flags & 2u) != 0u && c.a < pc.alpha_ref)
        discard;

    // Lightmap modulation (bit 2)
    if ((pc.flags & 4u) != 0u)
        c.rgb *= texture(u_lm, v_uv1).rgb;

    // Fog (factor computed per-vertex)
    if (pc.fog_params.x > 0.5)
        c.rgb = mix(pc.fog_colour.rgb, c.rgb, v_fog_factor);

    // Gamma correction
    c.rgb = pow(c.rgb, vec3(pc.inv_gamma));

    out_color = c;
}
