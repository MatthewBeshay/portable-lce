#version 450

layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec2 a_uv;
layout(location = 2) in vec4 a_color;
layout(location = 3) in vec4 a_normal;   // SNORM, .w unused

// std140 layout. nm_col* together encode mat3(modelview), each column
// padded to vec4. chunk_offset_lit packs chunk_offset.xyz + lit-flag.w.
// Lighting state is stored in eye-space (transformed by VulkanRenderPath
// at StateSetLightDirection time), matching the GL renderer.
layout(push_constant) uniform PushConstants {
    mat4 mvp;
    vec4 nm_col0;
    vec4 nm_col1;
    vec4 nm_col2;
    vec4 chunk_offset_lit;
    vec4 light0_dir;
    vec4 light1_dir;
    vec4 light_diffuse;
    vec4 light_ambient;
} pc;

layout(location = 0) out vec2  v_uv;
layout(location = 1) out vec4  v_color;
layout(location = 2) out float v_color_was_zero;
layout(location = 3) out float v_eye_dist;

void main() {
    vec3 world = a_pos + pc.chunk_offset_lit.xyz;
    vec4 clip  = pc.mvp * vec4(world, 1.0);
    clip.y = -clip.y;
    gl_Position = clip;
    // 2D texture transform (scale + offset), packed into the unused .w
    // lanes of the lighting block. Default is scale=(1,1), offset=(0,0)
    // = identity.
    vec2 tex_scale  = vec2(pc.nm_col0.w, pc.nm_col1.w);
    vec2 tex_offset = vec2(pc.nm_col2.w, pc.light0_dir.w);
    v_uv = a_uv * tex_scale + tex_offset;
    v_color = a_color.wzyx;
    // Strict 0x00000000 sentinel (matches GL renderer's vertex.vert).
    // The earlier loose `R+G+B < 0.004` check was misclassifying any
    // legitimately-dark vertex (e.g. low-brightness sky tint, dim faces)
    // as the sentinel and replacing them with whatever StateSetColour
    // happened to be set to that frame - which changes 20k+ times/sec
    // and explains the sky flashing / wrong colours.
    v_color_was_zero = (a_color == vec4(0.0)) ? 1.0 : 0.0;
    // Radial fog distance: length of eye-space position, matching GL's
    // `length(uMV * pos)`. Using clip.w gives PLANAR fog (flat boundary
    // perpendicular to view) which creates a visible horizontal line
    // cutting through the world. The modelview translation is packed
    // into light1_dir.w / light_diffuse.w / light_ambient.w.
    mat3 nm = mat3(pc.nm_col0.xyz, pc.nm_col1.xyz, pc.nm_col2.xyz);
    vec3 mv_translation = vec3(pc.light1_dir.w, pc.light_diffuse.w,
                               pc.light_ambient.w);
    vec3 eye_pos = nm * world + mv_translation;
    v_eye_dist = length(eye_pos);

    if (pc.chunk_offset_lit.w != 0.0) {
        vec3 raw_n = a_normal.xyz;
        // Tesselator can emit zero normals for stuff that didn't set one.
        // Skip the math in that case to leave the colour alone.
        if (dot(raw_n, raw_n) > 0.001) {
            vec3 n = normalize(nm * raw_n);
            float d0 = max(dot(n, pc.light0_dir.xyz), 0.0);
            float d1 = max(dot(n, pc.light1_dir.xyz), 0.0);
            v_color.rgb *=
                pc.light_ambient.xyz +
                pc.light_diffuse.xyz * clamp(d0 + d1, 0.0, 1.0);
        }
    }
}
