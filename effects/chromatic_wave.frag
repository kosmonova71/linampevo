#version 330 core
in vec2 v_texcoord;
out vec4 FragColor;

uniform sampler2D u_source;
uniform vec2 u_resolution;
uniform float u_time;
uniform float u_time_delta;
uniform float u_distort;
uniform float u_freq;
uniform float u_speed;
uniform float u_intensity;
uniform float iBass;
uniform float iTreb;

void main() {
    vec2 uv = v_texcoord;

    float w1 = sin((uv.x + uv.y) * u_freq * 6.28318 + u_time * (1.0 + u_speed * 2.0));
    float w2 = cos((uv.x - uv.y) * u_freq * 4.0 - u_time * (0.7 + u_speed));
    vec2 off = vec2(w1, w2) * u_distort * u_intensity * 0.02;

    float ca = u_intensity * (0.004 + 0.01 * u_distort) * (0.5 + 0.5 * iTreb);

    vec3 col;
    col.r = texture(u_source, uv + off + vec2(ca, 0.0)).r;
    col.g = texture(u_source, uv + off * 0.5).g;
    col.b = texture(u_source, uv + off - vec2(ca, 0.0)).b;

    float lum = dot(col, vec3(0.299, 0.587, 0.114));
    float shimmer = sin(uv.y * 20.0 + u_time * 3.0) * 0.5 + 0.5;
    col += vec3(0.05) * u_intensity * shimmer * smoothstep(0.4, 1.0, lum);
    col += vec3(0.03) * iBass * smoothstep(0.6, 1.0, lum);

    FragColor = vec4(col, 1.0);
}
