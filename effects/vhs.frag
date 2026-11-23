#version 330 core
in vec2 v_texcoord;
out vec4 FragColor;

uniform sampler2D u_source;
uniform vec2 u_resolution;
uniform float u_time;
uniform float u_time_delta;
uniform float u_glitch;
uniform float u_speed;
uniform float u_intensity;
uniform float u_threshold;
uniform float iBass;
uniform float iMid;
uniform float iTreb;

float hash(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

void main() {
    vec2 uv = v_texcoord;
    vec2 scene_uv = uv;

    float band_pos = 0.5 + 0.35 * sin(u_time * (0.3 + u_speed * 0.4) + u_glitch * 6.28318);
    float band = 1.0 - smoothstep(0.0, 0.10, abs(uv.y - band_pos));
    float tracking = band * (0.5 + u_glitch) * (0.6 + 0.4 * iMid);

    float row = floor(uv.y * u_resolution.y / 4.0);
    float jit = (hash(vec2(row, floor(u_time * 24.0))) - 0.5) * tracking * 0.08;

    float ca = (0.002 + u_glitch * 0.02 + 0.004 * iTreb) * (0.5 + 0.5 * u_intensity);
    vec2 dir = vec2(1.0, 0.35);

    vec3 col;
    col.r = texture(u_source, scene_uv + dir * (ca + jit)).r;
    col.g = texture(u_source, scene_uv + vec2(jit, 0.0)).g;
    col.b = texture(u_source, scene_uv - dir * (ca + jit)).b;

    float n = hash(uv * u_resolution + u_time * 61.0);
    col += (n - 0.5) * 0.12 * (0.3 + u_glitch + iBass * 0.4);

    col *= 0.9 + 0.1 * sin(uv.y * u_resolution.y * 1.5 + u_time * 8.0);
    col = mix(col, col * vec3(1.2, 0.9, 1.1) + vec3(0.05, 0.0, 0.08), tracking * 0.5);

    float head = (1.0 - smoothstep(0.0, 0.03, abs(uv.y - band_pos - 0.06))) * u_glitch;
    col += vec3(0.10) * head;

    float gate = smoothstep(u_threshold - 0.1, u_threshold + 0.1, u_glitch + iBass * 0.4);
    col *= 0.75 + 0.25 * gate;

    FragColor = vec4(col, 1.0);
}
