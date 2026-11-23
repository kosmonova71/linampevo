#version 330 core
in vec2 v_texcoord;
out vec4 FragColor;

uniform sampler2D u_source;
uniform sampler2D u_image;
uniform sampler2D u_image_next;
uniform vec2 u_resolution;
uniform float u_image_aspect;
uniform float u_image_blend;
uniform int u_segments;
uniform float u_angle;
uniform float u_speed;
uniform float u_intensity;
uniform float u_time;

void main() {
    vec2 uv = v_texcoord;
    vec4 scene = texture(u_source, uv);

    float sa = u_resolution.x / max(u_resolution.y, 1.0);
    float ia = max(u_image_aspect, 0.01);
    vec2 fit = vec2(min(1.0, sa / ia), min(1.0, ia / sa));
    vec2 iuv = (uv - 0.5) * fit + 0.5;

    // Kaleidoscope fold of the folder image around the centre.
    vec2 c = iuv - 0.5;
    float seg = float(max(u_segments, 2));
    float seg_ang = 6.28318 / seg;
    float rot = u_angle + u_time * 0.12 * (0.5 + u_speed);
    float r = length(c);
    float th = atan(c.y, c.x);
    th = mod(th - rot, seg_ang);
    if (th > seg_ang * 0.5) {
        th = seg_ang - th;
    }
    vec2 k = vec2(cos(th + rot), sin(th + rot)) * r + 0.5;

    vec4 img = mix(texture(u_image, k), texture(u_image_next, k), u_image_blend);
    float ki = clamp(u_intensity, 0.0, 1.0);
    vec3 col = mix(scene.rgb, img.rgb, ki * 0.8);
    col += vec3(0.02) * ki;
    FragColor = vec4(col, 1.0);
}
