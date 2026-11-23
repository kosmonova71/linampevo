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
uniform float u_angle;
uniform float u_distort;
uniform float iBass;
uniform float iMid;
uniform float iTreb;

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

vec2 hash22(vec2 p) {
    return vec2(hash21(p), hash21(p + 19.7));
}

void main() {
    vec2 uv = v_texcoord;
    vec4 scene = texture(u_source, uv);

    float aspect = u_resolution.x / max(u_resolution.y, 1.0);
    vec2 p = vec2(uv.x * aspect, uv.y) * u_scale * 8.0;

    float ca = cos(u_angle + u_time * u_speed * 0.1);
    float sa = sin(u_angle + u_time * u_speed * 0.1);
    p = mat2(ca, -sa, sa, ca) * p;

    float t = u_time * (0.4 + u_speed * 0.6);
    vec2 ip = floor(p);
    vec2 fp = fract(p);

    float f1 = 8.0;
    float f2 = 8.0;
    vec2 cell = vec2(0.0);
    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            vec2 g = vec2(float(i), float(j));
            vec2 o = hash22(ip + g);
            o = 0.5 + 0.5 * sin(t * 0.5 + 6.28318 * o);
            vec2 r = g + o - fp;
            float d = dot(r, r);
            if (d < f1) {
                f2 = f1;
                f1 = d;
                cell = ip + g;
            } else if (d < f2) {
                f2 = d;
            }
        }
    }

    float edge = sqrt(f2) - sqrt(f1);
    float cellid = hash21(cell);
    vec3 facet = 0.6 + 0.4 * cos(6.28318 * (cellid + vec3(0.0, 0.33, 0.67)));

    vec2 refr_dir = normalize(fp - 0.5 + 1e-4) * edge * u_distort * 0.05;
    vec2 refr_uv = clamp(uv + refr_dir, 0.0, 1.0);
    vec3 refr = texture(u_source, refr_uv).rgb;

    float line = 1.0 - smoothstep(0.0, 0.08, edge);
    vec3 col = refr * facet * 0.8 + vec3(line) * u_intensity * 1.2;
    col += facet * 0.1 * u_intensity;

    float k = clamp(u_intensity * (0.5 + 0.3 * iMid + 0.2 * iTreb), 0.0, 1.0);
    vec3 outc = mix(scene.rgb, col, k * 0.85);

    FragColor = vec4(outc, 1.0);
}
