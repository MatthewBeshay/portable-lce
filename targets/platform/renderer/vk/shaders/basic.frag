#version 450

layout(location = 0) in vec2  v_uv;
layout(location = 1) in vec4  v_color;
layout(location = 2) in float v_color_was_zero;
layout(location = 3) in float v_eye_dist;

layout(set = 0, binding = 0) uniform sampler2D u_tex;

layout(push_constant) uniform PC {
    layout(offset = 80)  uint  flags;        // bit 0 textured, bit 1 alpha_test
    layout(offset = 84)  float alpha_ref;
    layout(offset = 96)  vec4  state_colour;
    layout(offset = 112) vec4  fog_params;   // mode, start, end, density
    layout(offset = 128) vec4  fog_colour;
} pc;

layout(location = 0) out vec4 out_color;

void main() {
    vec4 c = (v_color_was_zero > 0.5) ? pc.state_colour : v_color;
    if ((pc.flags & 1u) != 0u) {
        out_color = texture(u_tex, v_uv) * c;
    } else {
        out_color = c;
    }
    if ((pc.flags & 2u) != 0u && out_color.a < pc.alpha_ref) {
        discard;
    }

    int mode = int(pc.fog_params.x);
    if (mode != 0) {
        float dist = v_eye_dist;
        float f = 1.0;
        if (mode == 1) {
            f = clamp((pc.fog_params.z - dist) /
                      max(pc.fog_params.z - pc.fog_params.y, 1e-4),
                      0.0, 1.0);
        } else if (mode == 2) {
            f = clamp(exp(-pc.fog_params.w * dist), 0.0, 1.0);
        } else if (mode == 3) {
            float d = pc.fog_params.w * dist;
            f = clamp(exp(-d * d), 0.0, 1.0);
        }
        out_color.rgb = mix(pc.fog_colour.rgb, out_color.rgb, f);
    }
}
