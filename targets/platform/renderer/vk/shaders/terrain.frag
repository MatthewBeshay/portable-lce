#version 450

layout(location = 0) in vec2  v_uv;
layout(location = 1) in vec4  v_color;
layout(location = 2) in float v_color_was_zero;

layout(set = 0, binding = 1) uniform sampler2D u_atlas;

layout(location = 0) out vec4 out_color;

void main() {
    // Chunks always sample the terrain atlas; sentinel-zero vertex colours
    // collapse to opaque white (the legacy Tesselator emits zero when no
    // per-vertex colour was set; for terrain that means "use the texture
    // as-is" rather than the global state colour).
    vec4 c = (v_color_was_zero > 0.5) ? vec4(1.0) : v_color;
    vec4 tex = texture(u_atlas, v_uv);
    if (tex.a < 0.1) discard;  // alpha cutout for grass / leaves / fences
    out_color = tex * c;
}
