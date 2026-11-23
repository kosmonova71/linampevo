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
uniform float u_zoom;
uniform float u_freq;
uniform float u_cx;
uniform float u_cy;
uniform float iBass;
uniform float iMid;
uniform float iTreb;

void main() {
    vec2 uv = v_texcoord;
    vec4 scene = texture(u_source, uv);

    float aspect = u_resolution.x / max(u_resolution.y, 1.0);
    vec2 p = uv - 0.5;
    p.x *= aspect;

    float horizon = 0.60 + 0.08 * sin(u_time * 0.4 + u_cx * 6.28318);
    float depth = clamp((uv.y - horizon) / max(0.001, 1.0 - uv.y), 0.0, 1.0);
    float persp = mix(0.02, 1.0, depth * depth * (0.5 + 0.5 * u_zoom));

    vec2 gp = vec2(p.x * persp, (uv.y - horizon) * persp);
    float freq = 8.0 + u_freq * 24.0 + u_scale * 8.0;
    vec2 g = abs(fract(gp * freq) - 0.5) / freq;
    float line = min(g.x, g.y);
    float glow = (1.0 - smoothstep(0.0, 0.06, line)) * (0.35 + 0.65 * iBass);

    float sky = smoothstep(horizon, 1.0, uv.y);
    vec3 skycol = mix(vec3(0.02, 0.0, 0.08), vec3(0.10, 0.02, 0.25), sky);
    vec3 gridcol = mix(vec3(1.0, 0.10, 0.60), vec3(0.10, 0.90, 1.00), clamp(uv.y * 2.0 + u_cy, 0.0, 1.0));

    vec3 col = skycol + gridcol * glow * u_intensity * 2.2;
    col *= 0.85 + 0.15 * sin(uv.y * u_resolution.y * 0.5);

    float k = clamp(u_intensity * (0.55 + 0.45 * iMid), 0.0, 1.0);
    vec3 outc = mix(scene.rgb, col, k * 0.85);
    outc += vec3(0.03) * iTreb * smoothstep(0.5, 1.0, uv.y);

    FragColor = vec4(outc, 1.0);
}
