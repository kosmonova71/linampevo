#version 330 core
in vec2 v_texcoord;
out vec4 FragColor;

uniform sampler2D u_source;
uniform sampler2D u_image;
uniform sampler2D u_image_next;
uniform vec2 u_resolution;
uniform float u_image_aspect;
uniform float u_image_blend;
uniform float u_time;
uniform float u_speed;
uniform float u_scale;
uniform float u_twist;
uniform float u_intensity;
uniform float u_mix;

void main() {
    vec2 uv = v_texcoord;
    vec4 scene = texture(u_source, uv);

    // Cover-fit the folder image into the frame aspect.
    float sa = u_resolution.x / max(u_resolution.y, 1.0);
    float ia = max(u_image_aspect, 0.01);
    vec2 fit = vec2(min(1.0, sa / ia), min(1.0, ia / sa));
    vec2 iuv = (uv - 0.5) * fit + 0.5;

    // Slow rotational swirl + breathing zoom around the centre.
    vec2 c = iuv - 0.5;
    float ang = u_twist * 6.28318 + u_time * (0.15 + 0.5 * u_speed);
    float cs = cos(ang);
    float sn = sin(ang);
    vec2 sw = vec2(c.x * cs - c.y * sn, c.x * sn + c.y * cs);
    float zoom = 0.7 + 0.9 * u_scale + 0.08 * sin(u_time * (0.4 + u_speed));
    iuv = sw / max(zoom, 0.05) + 0.5;

    vec4 img = mix(texture(u_image, iuv), texture(u_image_next, iuv), u_image_blend);
    img.rgb = pow(max(img.rgb, vec3(0.0)), vec3(0.85));

    float k = clamp(u_intensity * (0.55 + 0.45 * u_mix), 0.0, 1.0);
    vec3 col = mix(scene.rgb, img.rgb, k * 0.85);
    col += scene.rgb * 0.12 * (1.0 - k);
    FragColor = vec4(col, 1.0);
}
