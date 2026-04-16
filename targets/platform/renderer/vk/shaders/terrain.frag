#version 450

layout(location = 0) in vec2  v_uv;
layout(location = 1) in vec4  v_color;
layout(location = 2) in float v_color_was_zero;
layout(location = 3) in float v_eye_dist;
layout(location = 4) in vec2  v_lm_uv;

layout(set = 0, binding = 1) uniform sampler2D u_atlas;
layout(set = 0, binding = 2) uniform sampler2D u_lightmap;

layout(push_constant) uniform PC {
    layout(offset = 64) vec4 fog_params;   // mode, start, end, density
    layout(offset = 80) vec4 fog_colour;
    layout(offset = 96) vec4 tint;         // StateSetColour modulation
} pc;

layout(location = 0) out vec4 out_color;

void main() {
    vec4 c = (v_color_was_zero > 0.5) ? vec4(1.0) : v_color;
    vec4 tex = texture(u_atlas, v_uv);
    if (tex.a < 0.1) discard;
    vec4 lm = texture(u_lightmap, v_lm_uv);
    out_color = tex * c * vec4(lm.rgb, 1.0) * pc.tint;

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
