#version 450

layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec2 a_uv;
layout(location = 2) in vec4 a_color;
layout(location = 3) in vec4 a_normal;   // SNORM, .w unused

layout(push_constant) uniform PushConstants {
    mat4 mvp;
    vec3 chunk_offset;
    uint lit;
} pc;

layout(location = 0) out vec2  v_uv;
layout(location = 1) out vec4  v_color;
layout(location = 2) out float v_color_was_zero;
layout(location = 3) out float v_eye_dist;

// Hardcoded sun + sky directions in world space, Lambert shading.
// Replaces vs_4jcraft.sc's u_light0Dir / u_light1Dir until those state
// hooks are wired through StateSetLightDirection.
const vec3 SUN_DIR = normalize(vec3(0.5, 1.0, 0.3));
const vec3 SKY_DIR = normalize(vec3(-0.3, 1.0, -0.5));
const vec3 AMBIENT = vec3(0.5);
const vec3 DIFFUSE = vec3(0.5);

void main() {
    vec3 world = a_pos + pc.chunk_offset;
    vec4 clip  = pc.mvp * vec4(world, 1.0);
    clip.y = -clip.y;
    gl_Position = clip;
    v_uv = a_uv;
    v_color = a_color.wzyx;
    v_color_was_zero =
        ((v_color.r + v_color.g + v_color.b) < 0.004) ? 1.0 : 0.0;
    v_eye_dist = clip.w;

    if (pc.lit != 0u) {
        vec3 n = a_normal.xyz;
        // Tesselator can emit zero normals for stuff that didn't set one.
        // Skip the math in that case to leave the colour alone.
        if (dot(n, n) > 0.001) {
            float d0 = max(dot(n, SUN_DIR), 0.0);
            float d1 = max(dot(n, SKY_DIR), 0.0);
            v_color.rgb *= AMBIENT + DIFFUSE * clamp(d0 + d1, 0.0, 1.0);
        }
    }
}
