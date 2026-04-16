#version 450

// One indirect draw per visible chunk. The cull compute shader writes
// firstInstance = chunk slot index, which lands in gl_InstanceIndex here
// and lets us index ChunkMetadata for the chunk's world position.

layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec2 a_uv;
layout(location = 2) in vec4 a_color;
layout(location = 3) in vec4 a_normal;   // SNORM, .w unused

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
layout(location = 3) out float v_eye_dist;

const vec3 SUN_DIR = normalize(vec3(0.5, 1.0, 0.3));
const vec3 SKY_DIR = normalize(vec3(-0.3, 1.0, -0.5));
const vec3 AMBIENT = vec3(0.5);
const vec3 DIFFUSE = vec3(0.5);

void main() {
    vec3 world = meta.slots[gl_InstanceIndex].world_pos + a_pos;
    vec4 clip  = pc.mvp * vec4(world, 1.0);
    clip.y = -clip.y;
    gl_Position = clip;
    v_uv = a_uv;
    // Tesselator packs colour as (r<<24)|(g<<16)|(b<<8)|a; on little-endian
    // memory bytes are [a, b, g, r]. We bind that as R8G8B8A8_UNORM, so
    // a_color.x = a, .y = b, .z = g, .w = r. Swap back to RGBA.
    v_color = a_color.wzyx;
    v_color_was_zero =
        ((v_color.r + v_color.g + v_color.b) < 0.004) ? 1.0 : 0.0;
    v_eye_dist = clip.w;

    // Lambert shading using the packed normal. Always-on for chunks.
    vec3 n = a_normal.xyz;
    if (dot(n, n) > 0.001) {
        float d0 = max(dot(n, SUN_DIR), 0.0);
        float d1 = max(dot(n, SKY_DIR), 0.0);
        v_color.rgb *= AMBIENT + DIFFUSE * clamp(d0 + d1, 0.0, 1.0);
    }
}
