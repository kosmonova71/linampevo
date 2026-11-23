// evo/post/nonlinear_trails.glsl
// Nonlinear trails (trick 6): pull ghost trails out of the previous frame
// with a gain + gamma curve. Produces fire / smoke / plasma memory that
// lingers longer than the linear feedback mix alone.
vec3 postNonlinearTrails(vec3 col, vec2 uv, float t) {
    vec3 prev = texture(previous_frame, uv).rgb;
    vec3 ghost = pow(prev * 1.2, vec3(0.95));
    return col + ghost * 0.25;
}
