#version 450

// One indirect draw per visible chunk. The cull compute shader writes
// firstInstance = chunk slot index, which lands in gl_InstanceIndex here
// and lets us index ChunkMetadata for the chunk's world position.

layout(location = 0) in vec3  a_pos;
layout(location = 1) in vec2  a_uv;
layout(location = 2) in vec4  a_color;
layout(location = 3) in vec4  a_normal;     // SNORM, .w unused
layout(location = 4) in ivec2 a_lm_raw;     // packed int16 lightmap coords

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
layout(location = 4) out vec2  v_lm_uv;

void main() {
    vec3 world = meta.slots[gl_InstanceIndex].world_pos + a_pos;
    vec4 clip  = pc.mvp * vec4(world, 1.0);
    clip.y = -clip.y;
    gl_Position = clip;
    v_uv = a_uv;
    // Tesselator packs colour as (r<<24)|(g<<16)|(b<<8)|a; on little-endian
    // memory bytes are [a, b, g, r]. R8G8B8A8_UNORM gives us those bytes
    // as (a, b, g, r) -- swap back to RGBA.
    v_color = a_color.wzyx;
    v_color_was_zero =
        ((v_color.r + v_color.g + v_color.b) < 0.004) ? 1.0 : 0.0;
    v_eye_dist = clip.w;

    // Lightmap UV. Tesselator writes 0xfe00fe00 (= int16 -512, -512) when
    // a vertex has no per-vertex lightmap coord; the legacy GL renderer
    // falls back to a global UV in that case. Until StateSetVertexTextureUV
    // is wired up, treat the sentinel as fully lit (15/16 of the 16x16
    // lightmap = full sky + full block light).
    if (a_lm_raw.x <= -2 || a_lm_raw.y <= -2) {
        v_lm_uv = vec2(15.0 / 16.0);
    } else {
        // The +8 byte offset packed by Tesselator already lands us on a
        // texel centre; just scale to 0..1.
        v_lm_uv = vec2(a_lm_raw) / 256.0;
    }

    // No directional lighting here -- the lightmap and per-vertex AO from
    // Tesselator carry the shading. Adding sun/sky on top double-darkens
    // back-facing geometry and washes out the lightmap signal.
}
