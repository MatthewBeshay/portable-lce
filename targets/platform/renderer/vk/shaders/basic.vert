#version 450
#extension GL_GOOGLE_include_directive : require

#include "common.glsl"

layout(location = 0) in vec3  a_pos;
layout(location = 1) in vec2  a_uv;
layout(location = 2) in vec4  a_color;    // R8G8B8A8_UNORM
layout(location = 3) in vec4  a_normal;   // R8G8B8A8_SNORM — maps [-128,127] to [-1,1]
layout(location = 4) in ivec2 a_lm_raw;   // R16G16_SINT — lightmap UVs or sentinel

layout(location = 0) out vec2  v_uv;
layout(location = 1) out vec4  v_color;
layout(location = 2) out float v_fog_factor;
layout(location = 3) out vec2  v_uv1;       // lightmap UV

void main() {
    vec3 world = a_pos + pc.chunk_lit.xyz;

    // Clip-space position. Viewport has negative height — Y-flip in rasterizer.
    gl_Position = pc.mvp * vec4(world, 1.0);

    // Texture UV with scale + offset transform
    vec2 tex_scale  = vec2(pc.nm0.w, pc.nm1.w);
    vec2 tex_offset = vec2(pc.nm2.w, pc.tex_mv.x);
    v_uv = a_uv * tex_scale + tex_offset;

    // Lightmap UV — raw int16 coords divided by 256 for 0-1 range.
    // Sentinel: a_lm_raw.x <= -500 means use global lightmap fallback.
    {
        vec2 lm;
        if (a_lm_raw.x <= -500) {
            lm = vec2(float(frame.global_lm_packed & 0xFFFFu),
                       float(frame.global_lm_packed >> 16u));
        } else {
            lm = vec2(a_lm_raw);
        }
        v_uv1 = lm / 256.0;
    }

    // Vertex colour — RGBA packing, no swizzle needed.
    // Sentinel: RGB all zero = use state_colour (base tint).
    bool sentinel = all(equal(a_color.rgb, vec3(0.0)));
    vec4 col = sentinel ? pc.state_colour : a_color;

    // Per-vertex directional lighting (2 lights + ambient)
    if (pc.chunk_lit.w > 0.5) {
        vec3 raw_n = a_normal.xyz;  // SNORM: already in [-1,1]
        if (dot(raw_n, raw_n) > 0.001) {
            mat3 nm = mat3(pc.nm0.xyz, pc.nm1.xyz, pc.nm2.xyz);
            vec3 n = normalize(nm * raw_n);
            float d0 = max(dot(n, frame.light0_dir.xyz), 0.0);
            float d1 = max(dot(n, frame.light1_dir.xyz), 0.0);
            col.rgb *= frame.light_ambient.xyz +
                       frame.light_diffuse.xyz * clamp(d0 + d1, 0.0, 1.0);
        }
    }
    v_color = col;

    // Per-vertex radial fog — length(MV * pos) for spherical boundary
    mat3 nm = mat3(pc.nm0.xyz, pc.nm1.xyz, pc.nm2.xyz);
    vec3 mv_t = pc.tex_mv.yzw;
    float e_dist = length(nm * world + mv_t);

    int fog_mode = int(frame.fog_params.x);
    if (fog_mode == 1)
        v_fog_factor = clamp((frame.fog_params.z - e_dist) /
                             max(frame.fog_params.z - frame.fog_params.y, 1e-4),
                             0.0, 1.0);
    else if (fog_mode == 2)
        v_fog_factor = clamp(exp(-frame.fog_params.w * e_dist), 0.0, 1.0);
    else if (fog_mode == 3) {
        float d = frame.fog_params.w * e_dist;
        v_fog_factor = clamp(exp(-d * d), 0.0, 1.0);
    } else
        v_fog_factor = 1.0;
}
