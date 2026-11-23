// evo/post/chromatic_aberration.glsl
// Chromatic aberration: split RGB along the radial direction by resampling
// the previous frame (previous_frame is provided by the assembler). Reads as a
// lens / prism edge fringing.
vec3 postChromaticAberration(vec3 col, vec2 uv, float t) {
    vec2 dir = uv - 0.5;
    float amt = 0.003 * (1.0 + bass * 2.0) * (0.6 + 0.4 * sin(t * 0.2));
    float r = texture(previous_frame, uv + dir * amt).r;
    float b = texture(previous_frame, uv - dir * amt).b;
    return vec3(r, col.g, b);
}
