#version 450

layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec2 a_uv;
layout(location = 2) in vec4 a_color;

layout(push_constant) uniform PushConstants {
    mat4 mvp;
    vec3 chunk_offset;
} pc;

layout(location = 0) out vec2  v_uv;
layout(location = 1) out vec4  v_color;
layout(location = 2) out float v_color_was_zero;

void main() {
    // Chunks submit vertices in chunk-local space; add the per-draw offset
    // to lift them into the world.
    vec4 clip = pc.mvp * vec4(a_pos + pc.chunk_offset, 1.0);
    // Vulkan NDC has Y pointing down vs OpenGL's Y-up - flip post-transform.
    clip.y = -clip.y;
    gl_Position = clip;
    v_uv = a_uv;
    // Tesselator packs colour as (r<<24)|(g<<16)|(b<<8)|a in a uint32, which
    // on little-endian lands as bytes [a, b, g, r]. We bind it as
    // R8G8B8A8_UNORM, so a_color.x = a, .y = b, .z = g, .w = r. Swap back.
    v_color = a_color.wzyx;
    // Sentinel: bytes all zero means "use the GL-state colour fallback".
    v_color_was_zero = (a_color == vec4(0.0)) ? 1.0 : 0.0;
}
