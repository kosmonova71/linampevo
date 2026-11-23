#version 330 core
in vec2 v_texcoord;
out vec4 FragColor;

uniform sampler2D u_source;
uniform vec2 u_resolution;
uniform float u_time;
uniform float u_time_delta;
uniform float u_warp;
uniform float u_scale;
uniform float u_speed;
uniform float u_intensity;
uniform float u_distort;
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

float fbm(vec2 p) {
    float v = 0.0;
    float a = 0.5;
    for (int i = 0; i < 3; i++) {
        v += a * noise(p);
        p = p * 2.03 + vec2(11.3, 7.7);
        a *= 0.55;
    }
    return v;
}

void main() {
    vec2 uv = v_texcoord;
    vec4 scene = texture(u_source, uv);

    float aspect = u_resolution.x / max(u_resolution.y, 1.0);
    vec2 p = vec2(uv.x * aspect, uv.y) * u_scale * 2.0;
    float t = u_time * (0.2 + u_speed * 0.5);

    vec2 q = vec2(fbm(p + vec2(0.0, t)), fbm(p + vec2(5.2, 1.3) - t));
    vec2 rr = vec2(
        fbm(p + q * (1.5 + u_warp * 2.0) + vec2(1.7, 9.2) + t * 0.4),
        fbm(p + q * (1.5 + u_warp * 2.0) + vec2(8.3, 2.8) - t * 0.3)
    );
    float f = fbm(p * 1.3 + rr * (1.0 + u_distort * 2.0) + vec2(t * 0.2, -t * 0.1));

    vec3 base = vec3(0.05, 0.07, 0.12);
    vec3 silver = vec3(0.75, 0.80, 0.90);
    vec3 blue = vec3(0.20, 0.50, 1.00);
    vec3 colr = mix(base, silver, smoothstep(0.2, 0.7, f));
    colr = mix(colr, blue, smoothstep(0.55, 0.95, f) * 0.7);
    colr += vec3(0.10, 0.15, 0.25) * sin(f * 20.0 + t * 2.0) * u_intensity;

    float k = clamp(u_intensity * (0.5 + 0.5 * iMid), 0.0, 1.0);
    vec3 outc = mix(scene.rgb, colr, k * 0.8);
    outc += vec3(0.02) * iTreb * smoothstep(0.5, 1.0, f);

    FragColor = vec4(outc, 1.0);
}
