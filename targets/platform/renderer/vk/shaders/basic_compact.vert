#version 450
#extension GL_GOOGLE_include_directive : require

#include "common.glsl"

// Compact 16-byte vertex format — 8x int16.
// Decoded in-shader to avoid a per-draw CPU expansion pass.
//
// Layout:
//   int16 pos.x (units of 1/1024)
//   int16 pos.y
//   int16 pos.z
//   int16 color_5_6_5 (biased by +32768)
//   int16 uv.x (units of 1/8192)
//   int16 uv.y
//   int16 lightmap.u
//   int16 lightmap.v
layout(location = 0) in ivec4 a_packed0;  // (pos.x, pos.y, pos.z, color_5_6_5)
layout(location = 1) in ivec4 a_packed1;  // (uv.x, uv.y, lm.u, lm.v)

layout(location = 0) out vec2  v_uv;
layout(location = 1) out vec4  v_color;
layout(location = 2) out float v_fog_factor;
layout(location = 3) out vec2  v_uv1;

void main() {
    // Decode compact inputs.
    vec3 a_pos  = vec3(a_packed0.xyz) / 1024.0;
    vec2 a_uv   = vec2(a_packed1.xy) / 8192.0;

    // Color: int16 shifted by +32768 to get an unsigned 16-bit field packed 5-6-5.
    uint packed = uint(a_packed0.w + 32768);
    vec4 a_color = vec4(
        float(packed        & 0x1Fu) / 31.0,
        float((packed >> 5u) & 0x3Fu) / 63.0,
        float((packed >> 11u) & 0x1Fu) / 31.0,
        1.0);

    // The compact format has no per-vertex normal — use +Y so per-vertex
    // lighting still produces a sensible top-lit result when enabled.
    vec3 raw_n = vec3(0.0, 1.0, 0.0);

    ivec2 a_lm_raw = a_packed1.zw;

    vec3 world = a_pos + pc.chunk_lit.xyz;

    // Clip-space position. Viewport has negative height — Y-flip in rasterizer.
    gl_Position = pc.mvp * vec4(world, 1.0);

    // Texture UV transform
    vec2 tex_scale  = vec2(pc.nm0.w, pc.nm1.w);
    vec2 tex_offset = vec2(pc.nm2.w, pc.tex_mv.x);
    v_uv = a_uv * tex_scale + tex_offset;

    // Lightmap UV
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

    // Vertex color — sentinel (RGB all zero) means "untinted, use the
    // state_colour push-constant". In the compact RGB565 encoding this
    // corresponds to the producer emitting int16(-32768) as the colour
    // field (all colour bits cleared after the +32768 bias). Chunk mesh
    // builders reserve that encoding for the untinted path.
    bool sentinel = all(equal(a_color.rgb, vec3(0.0)));
    vec4 col = sentinel ? pc.state_colour : a_color;

    // Per-vertex directional lighting
    if (pc.chunk_lit.w > 0.5 && dot(raw_n, raw_n) > 0.001) {
        mat3 nm = mat3(pc.nm0.xyz, pc.nm1.xyz, pc.nm2.xyz);
        vec3 n = normalize(nm * raw_n);
        float d0 = max(dot(n, frame.light0_dir.xyz), 0.0);
        float d1 = max(dot(n, frame.light1_dir.xyz), 0.0);
        col.rgb *= frame.light_ambient.xyz +
                   frame.light_diffuse.xyz * clamp(d0 + d1, 0.0, 1.0);
    }
    v_color = col;

    // Per-vertex radial fog
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
