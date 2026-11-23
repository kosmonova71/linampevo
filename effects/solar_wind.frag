#version 330 core
in vec2 v_texcoord;
out vec4 FragColor;

uniform sampler2D u_source;
uniform vec2 u_resolution;
uniform float u_time;
uniform float u_time_delta;
uniform float u_speed;
uniform float u_radius;
uniform float u_intensity;
uniform float u_scale;
uniform float u_freq;
uniform float iBass;
uniform float iMid;
uniform float iTreb;

float hash(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

void main() {
    vec2 uv = v_texcoord;
    vec4 scene = texture(u_source, uv);

    float aspect = u_resolution.x / max(u_resolution.y, 1.0);
    vec2 p = (uv - 0.5) * vec2(aspect, 1.0);
    float r = length(p);
    float ang = atan(p.y, p.x);
    float t = u_time * (0.3 + u_speed * 0.7);

    float streak = noise(vec2(ang * (0.5 + u_freq * 0.4) + r * 2.0, r * u_scale * 6.0 - t * 2.0));
    streak = smoothstep(0.55, 0.95, streak);

    float radial = smoothstep(0.05, 0.4, r) * (1.0 - smoothstep(0.6, 1.2, r));
    float pulse = 0.5 + 0.5 * sin(t * 3.0 - r * 8.0);
    vec3 wind = mix(vec3(1.0, 0.55, 0.15), vec3(1.0, 0.9, 0.6), streak) * (0.4 + 0.6 * pulse);

    float amt = streak * radial * u_intensity * (0.5 + 0.8 * iBass);
    vec3 col = scene.rgb + wind * amt * 1.5;

    float core = (1.0 - smoothstep(0.0, 0.35, r)) * u_intensity * (0.4 + 0.6 * iBass);
    col += vec3(1.0, 0.7, 0.3) * core * 0.4;

    FragColor = vec4(col, 1.0);
}
