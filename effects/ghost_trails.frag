#version 330 core
in vec2 v_texcoord;
out vec4 FragColor;

uniform sampler2D u_source;
uniform sampler2D u_previous_frame;
uniform vec2 u_resolution;
uniform float u_time;
uniform float u_time_delta;
uniform float u_trail;
uniform float u_decay;
uniform float u_speed;
uniform float u_intensity;
uniform float iBass;
uniform float iMid;
uniform float iTreb;

void main() {
    vec2 uv = v_texcoord;
    vec4 cur = texture(u_source, uv);

    vec2 drift = vec2(
        sin(u_time * (0.6 + u_speed) * 2.0),
        cos(u_time * (0.5 + u_speed) * 1.7)
    ) * u_intensity * 0.012;
    vec4 prev = texture(u_previous_frame, uv - drift);

    float keep = clamp(u_trail * u_decay, 0.0, 0.95);
    vec3 col = cur.rgb * (1.0 - keep * 0.7) + prev.rgb * keep;
    col += cur.rgb * iBass * 0.25;

    float energy = 0.6 + 0.4 * iMid + 0.3 * iTreb;
    col = clamp(col * energy, 0.0, 1.4);

    FragColor = vec4(col, 1.0);
}
