#version 330 core
in vec2 v_uv;
out vec4 FragColor;
uniform sampler2D u_from;
uniform sampler2D u_to;
uniform float u_progress;
uniform vec2 u_resolution;

float rand(vec2 co) {
    return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
    vec2 pixel_uv = floor(v_uv * u_resolution) / u_resolution;
    FragColor = mix(texture(u_from, v_uv), texture(u_to, v_uv), step(rand(pixel_uv), u_progress));
}