#version 330 core
in vec2 v_texcoord;
out vec4 FragColor;

uniform sampler2D u_source;
uniform vec2 u_resolution;
uniform float u_time;
uniform float u_time_delta;
uniform float u_speed;
uniform float u_zoom;
uniform float u_intensity;
uniform float u_scale;
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
    float t = u_time * (0.5 + u_speed * 0.8);

    vec2 sp = vec2(ang * 3.0 / (0.2 + r * 2.0), r * 8.0 - t * (1.0 + u_speed * 2.0));
    float n = noise(sp * vec2(6.0, 1.0) * u_scale);
    float star = smoothstep(0.90, 1.0, n);
    float streak = smoothstep(0.78, 1.0, noise(vec2(ang * 2.0, r * 4.0 - t * 3.0) * u_scale));

    vec3 starcol = mix(vec3(0.6, 0.8, 1.0), vec3(1.0, 0.9, 0.7), fract(n * 7.0));
    float fall = 1.0 - smoothstep(0.15, 1.0, r);
    float amt = (star * 0.7 + streak * 0.5) * fall * u_intensity * (0.6 + 0.6 * iMid);

    vec3 col = scene.rgb + starcol * amt * 2.0;
    col += vec3(0.4, 0.6, 1.0) * (1.0 - smoothstep(0.0, 0.4, r)) * u_intensity * 0.3 * (0.5 + 0.5 * iTreb);

    FragColor = vec4(col, 1.0);
}
