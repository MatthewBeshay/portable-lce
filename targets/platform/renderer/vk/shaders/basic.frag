#version 450

layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec4 v_color;
layout(location = 2) in float v_color_was_zero;

layout(set = 0, binding = 0) uniform sampler2D u_tex;

layout(push_constant) uniform PC {
    layout(offset = 80) uint  textured;
    layout(offset = 96) vec4  state_colour;
} pc;

layout(location = 0) out vec4 out_color;

void main() {
    // Tesselator writes 0x00000000 as a sentinel meaning "use the GL-state
    // colour set via StateSetColour" instead of the per-vertex colour.
    vec4 c = (v_color_was_zero > 0.5) ? pc.state_colour : v_color;
    if (pc.textured != 0u) {
        out_color = texture(u_tex, v_uv) * c;
    } else {
        out_color = c;
    }
}
