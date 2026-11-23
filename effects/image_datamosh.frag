#version 330 core
in vec2 v_texcoord;
out vec4 FragColor;

uniform sampler2D u_source;
uniform sampler2D u_image;
uniform sampler2D u_image_next;
uniform vec2 u_resolution;
uniform float u_image_aspect;
uniform float u_image_blend;
uniform float u_image_index;
uniform float u_glitch;
uniform float u_speed;
uniform float iBass;
uniform float u_time;

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

void main() {
    vec2 uv = v_texcoord;
    vec4 scene = texture(u_source, uv);

    float sa = u_resolution.x / max(u_resolution.y, 1.0);
    float ia = max(u_image_aspect, 0.01);
    vec2 fit = vec2(min(1.0, sa / ia), min(1.0, ia / sa));
    vec2 iuv = (uv - 0.5) * fit + 0.5;

    // Datamosh-style slice displacement, driven by glitch gene + bass.
    float drive = clamp(u_glitch * (0.4 + 0.6 * iBass), 0.0, 1.0);
    float slice = mix(0.12, 0.015, drive);
    float row = floor(iuv.y / max(slice, 0.004));
    float t = floor(u_time * (2.0 + 8.0 * u_speed)) + u_image_index * 17.0;
    float shift = (hash(vec2(row, t)) - 0.5) * drive * 0.35;
    iuv.x = fract(iuv.x + shift);

    vec4 a = texture(u_image, iuv);
    vec4 b = texture(u_image_next, iuv + vec2(shift * 0.5, 0.0));
    vec4 img = mix(a, b, u_image_blend);

    // RGB channel split proportional to drive.
    vec4 r = texture(u_image, iuv + vec2(drive * 0.012, 0.0));
    vec4 c2 = texture(u_image, iuv - vec2(drive * 0.012, 0.0));
    img.r = mix(img.r, r.r, drive);
    img.b = mix(img.b, c2.b, drive);

    float k = clamp(0.35 + 0.65 * drive, 0.0, 1.0);
    vec3 col = mix(scene.rgb, img.rgb, k * 0.85);
    FragColor = vec4(col, 1.0);
}
