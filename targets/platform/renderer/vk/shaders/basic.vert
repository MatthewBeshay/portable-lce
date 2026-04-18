#version 450

layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec2 a_uv;
layout(location = 2) in vec4 a_color;    // R8G8B8A8_UNORM
layout(location = 3) in vec4 a_normal;   // R8G8B8A8_SNORM

// Shared push-constant block (256 bytes, visible to both stages).
// The .w lanes of the lighting vectors carry auxiliary data to stay
// within the 256-byte push-constant budget.
layout(push_constant) uniform PC {
    mat4 mvp;               // 0
    vec4 nm_col0;           // 64   mat3(mv)[0].xyz, tex_scale_x
    vec4 nm_col1;           // 80   mat3(mv)[1].xyz, tex_scale_y
    vec4 nm_col2;           // 96   mat3(mv)[2].xyz, tex_offset_x
    vec4 chunk_lit;         // 112  chunk_offset.xyz, lighting_enabled
    vec4 l0_tex;            // 128  light0_dir.xyz, tex_offset_y
    vec4 l1_mv;             // 144  light1_dir.xyz, mv_translation.x
    vec4 ldiff_mv;          // 160  light_diffuse.xyz, mv_translation.y
    vec4 lamb_mv;           // 176  light_ambient.xyz, mv_translation.z
    vec4 fog_params;        // 192  mode, start, end, density
    vec4 state_colour;      // 208
    vec4 fog_colour;        // 224
    float alpha_ref;        // 240
    float inv_gamma;        // 244
    uint flags;             // 248
    uint _pad;              // 252
} pc;

layout(location = 0) out vec2  v_uv;
layout(location = 1) out vec4  v_color;
layout(location = 2) out float v_fog_factor;

void main() {
    vec3 world = a_pos + pc.chunk_lit.xyz;

    // Clip-space position (Y-flip baked into mvp by CPU)
    gl_Position = pc.mvp * vec4(world, 1.0);

    // Texture UV: scale + offset (covers identity / atlas transforms)
    vec2 tex_scale  = vec2(pc.nm_col0.w, pc.nm_col1.w);
    vec2 tex_offset = vec2(pc.nm_col2.w, pc.l0_tex.w);
    v_uv = a_uv * tex_scale + tex_offset;

    // Vertex colour — the Tesselator packs RGBA as (R<<24|G<<16|B<<8|A).
    // On little-endian that stores bytes [A,B,G,R]. R8G8B8A8_UNORM reads
    // those bytes into (R,G,B,A) = (A,B,G,R). The .abgr (.wzyx) swizzle
    // restores the correct (R,G,B,A) order. Matches GL's aColor.abgr.
    //
    // Sentinel: all-zero vertex colour means "use state_colour instead".
    // Exact comparison matches GL renderer (avoids false positives on
    // legitimately dark vertices).
    bool sentinel = all(equal(a_color, vec4(0.0)));
    vec4 col = sentinel ? pc.state_colour : a_color.wzyx;

    // Per-vertex lighting (two directional lights + ambient)
    if (pc.chunk_lit.w > 0.5) {
        vec3 raw_n = a_normal.xyz;
        // Skip lighting for vertices with no normal set (zero-length)
        if (dot(raw_n, raw_n) > 0.001) {
            mat3 nm = mat3(pc.nm_col0.xyz, pc.nm_col1.xyz, pc.nm_col2.xyz);
            vec3 n = normalize(nm * raw_n);
            float d0 = max(dot(n, pc.l0_tex.xyz), 0.0);
            float d1 = max(dot(n, pc.l1_mv.xyz), 0.0);
            col.rgb *= pc.lamb_mv.xyz +
                       pc.ldiff_mv.xyz * clamp(d0 + d1, 0.0, 1.0);
        }
    }
    v_color = col;

    // Radial fog distance — length(MV * pos) for spherical fog boundary.
    // MV = mat3 rotation | translation, packed into the .w lanes.
    mat3 nm = mat3(pc.nm_col0.xyz, pc.nm_col1.xyz, pc.nm_col2.xyz);
    vec3 mv_t = vec3(pc.l1_mv.w, pc.ldiff_mv.w, pc.lamb_mv.w);
    float e_dist = length(nm * world + mv_t);

    int fog_mode = int(pc.fog_params.x);
    if (fog_mode == 1) {
        // Linear
        v_fog_factor = clamp((pc.fog_params.z - e_dist) /
                             max(pc.fog_params.z - pc.fog_params.y, 1e-4),
                             0.0, 1.0);
    } else if (fog_mode == 2) {
        // Exponential
        v_fog_factor = clamp(exp(-pc.fog_params.w * e_dist), 0.0, 1.0);
    } else if (fog_mode == 3) {
        // Exponential squared
        float d = pc.fog_params.w * e_dist;
        v_fog_factor = clamp(exp(-d * d), 0.0, 1.0);
    } else {
        v_fog_factor = 1.0;
    }
}
