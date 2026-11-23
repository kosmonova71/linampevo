// evo/post/polar_galaxy.glsl
// Polar domain distortion (trick 2): twist the feedback sample in polar
// space to grow liquid galaxies, spiral organisms and vortex flowers.
vec3 postPolarGalaxy(vec3 col, vec2 uv, float t) {
    vec2 c = uv - 0.5;
    float r = length(c);
    float th = atan(c.y, c.x);
    th += sin(r * 8.0 - t * 0.5) * 0.3;
    th += sin(th * 6.0 + t * 0.3) * 0.15;
    vec2 q = vec2(cos(th), sin(th)) * r + 0.5;
    vec3 swirl = texture(previous_frame, q).rgb;
    return mix(col, swirl, 0.35);
}
