#version 450

// One indirect draw per visible chunk. The cull compute shader writes
// firstInstance = chunk slot index, which lands in gl_InstanceIndex here
// and lets us index ChunkMetadata for the chunk's world position.

layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec2 a_uv;
layout(location = 2) in vec4 a_color;

struct ChunkSlot {
    vec3 world_pos;
    uint vert_count;
    vec3 aabb_min;
    uint base_vertex;
    vec3 aabb_max;
    uint face_mask;
};

layout(set = 0, binding = 0, std430) readonly buffer ChunkMeta {
    ChunkSlot slots[];
} meta;

layout(push_constant) uniform PC {
    mat4 mvp;
} pc;

layout(location = 0) out vec2  v_uv;
layout(location = 1) out vec4  v_color;
layout(location = 2) out float v_color_was_zero;

void main() {
    vec3 world = meta.slots[gl_InstanceIndex].world_pos + a_pos;
    vec4 clip  = pc.mvp * vec4(world, 1.0);
    clip.y = -clip.y;  // Vulkan NDC Y-down vs OpenGL Y-up
    gl_Position = clip;
    v_uv = a_uv;
    // Tesselator packs colour as (r<<24)|(g<<16)|(b<<8)|a; on little-endian
    // memory bytes are [a, b, g, r]. With R8G8B8A8_UNORM that becomes
    // a_color = (a, b, g, r) - swap back to RGBA.
    v_color = a_color.wzyx;
    v_color_was_zero = (a_color == vec4(0.0)) ? 1.0 : 0.0;
}
