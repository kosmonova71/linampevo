#version 330 core
in vec2 v_texcoord;
out vec4 FragColor;

uniform sampler2D u_source;
uniform vec2 u_resolution;
uniform float u_time;
uniform float u_time_delta;
uniform float u_angle;
uniform float u_radius;
uniform float u_speed;
uniform float u_intensity;
uniform float u_distort;
uniform float iBass;
uniform float iTreb;

void main() {
    vec2 uv = v_texcoord;
    float aspect = u_resolution.x / max(u_resolution.y, 1.0);
    vec2 p = uv - 0.5;
    p.x *= aspect;

    float r = length(p);
    float ang = atan(p.y, p.x);
    float t = u_time * (0.4 + u_speed * 0.8);

    float swirl = u_distort * 0.6 * (1.0 - smoothstep(0.0, 0.8, r)) + u_radius * 0.3;
    float a2 = ang + swirl * (0.5 + r * 2.0) + t * 0.3 + u_angle;

    vec2 suv = vec2(
        0.5 + cos(a2) * r / aspect,
        0.5 + sin(a2) * r
    );
    suv = clamp(suv, 0.0, 1.0);
    vec3 base = texture(u_source, suv).rgb;

    float rings = sin(r * 30.0 - t * 5.0 + u_angle);
    float ring = smoothstep(0.6, 1.0, rings);
    vec3 ringcol = 0.5 + 0.5 * cos(6.28318 * (r * 3.0 - t * 0.3 + vec3(0.0, 0.33, 0.67)));

    vec3 col = base + ringcol * ring * u_intensity * 0.8 * (0.5 + 0.5 * iTreb);
    col += ringcol * (1.0 - smoothstep(0.0, 0.7, r)) * u_radius * 0.4 * (0.6 + 0.4 * iBass);

    FragColor = vec4(col, 1.0);
}
