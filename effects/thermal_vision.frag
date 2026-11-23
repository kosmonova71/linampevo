#version 330 core
in vec2 v_texcoord;
out vec4 FragColor;

uniform sampler2D u_source;
uniform vec2 u_resolution;
uniform float u_time;
uniform float u_time_delta;
uniform float u_intensity;
uniform float u_scale;
uniform float u_speed;
uniform float u_threshold;
uniform float iMid;
uniform float iBass;

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

    float lum = dot(scene.rgb, vec3(0.299, 0.587, 0.114));
    float n = noise(uv * u_resolution * 0.25 + u_time * u_speed);
    float heat = clamp(lum * 1.4 + (n - 0.5) * 0.3 * u_scale, 0.0, 1.0);
    heat = smoothstep(u_threshold, u_threshold + 0.4, heat);

    vec3 cold = vec3(0.02, 0.05, 0.15);
    vec3 midc = vec3(0.90, 0.35, 0.10);
    vec3 hot = vec3(1.00, 0.90, 0.60);
    vec3 pal = mix(cold, midc, smoothstep(0.0, 0.55, heat));
    pal = mix(pal, hot, smoothstep(0.55, 1.0, heat));

    float contour = abs(fract(heat * 8.0) - 0.5);
    pal += vec3(0.10) * (1.0 - smoothstep(0.0, 0.08, contour)) * u_intensity;
    pal *= 0.85 + 0.3 * iBass;

    float k = clamp(u_intensity * (0.6 + 0.4 * iMid), 0.0, 1.0);
    vec3 outc = mix(scene.rgb, pal, k * 0.85);

    FragColor = vec4(outc, 1.0);
}
