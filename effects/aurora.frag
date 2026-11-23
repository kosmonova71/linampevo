#version 330 core
in vec2 v_texcoord;
out vec4 FragColor;

uniform sampler2D u_source;
uniform vec2 u_resolution;
uniform float u_time;
uniform float u_time_delta;
uniform float u_speed;
uniform float u_scale;
uniform float u_intensity;
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

vec3 palette(float x) {
    return 0.5 + 0.5 * cos(6.28318 * (x + vec3(0.0, 0.33, 0.67)));
}

void main() {
    vec2 uv = v_texcoord;
    vec4 scene = texture(u_source, uv);

    float aspect = u_resolution.x / max(u_resolution.y, 1.0);
    vec2 p = vec2(uv.x * aspect, uv.y) * u_scale * 3.0;
    float t = u_time * (0.15 + u_speed * 0.55);

    vec2 q = vec2(fbm(p + vec2(0.0, t * 0.35)), fbm(p + vec2(3.1, 1.7) - t * 0.25));
    vec2 r = vec2(
        fbm(p + q * 1.6 + vec2(1.3, 9.1) + t * 0.45),
        fbm(p + q * 1.6 + vec2(8.9, 2.4) - t * 0.30)
    );
    float f = fbm(p + r * 2.2 + vec2(t * 0.1, -t * 0.07));

    float band = noise(vec2(p.x * 0.8, p.y * 2.5 + f * 2.5 - t * 0.7));
    float glow = smoothstep(0.35, 0.95, band) * u_intensity;
    glow *= 0.55 + 0.45 * iMid + 0.25 * iBass;

    vec3 aur = palette(f * 1.4 + t * 0.05 + iTreb * 0.2);
    float k = clamp(glow * 1.2, 0.0, 1.0);
    vec3 col = mix(scene.rgb, scene.rgb * 0.75 + aur * glow * 1.4, k);
    col += vec3(0.02) * iTreb * smoothstep(0.6, 1.0, uv.y);

    FragColor = vec4(col, 1.0);
}
