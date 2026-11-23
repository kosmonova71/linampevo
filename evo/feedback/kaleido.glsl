// evo/feedback/kaleido.glsl
// Kaleidoscope feedback recursion (trick 1): fold the feedback sample UV
// through an n-fold mirror symmetry before reading the previous frame. Combined
// with the organism's own symmetry this produces fractal mirrors, crystal
// structures and mandala-like evolution across the feedback buffer.
// Selected via genome.feedback_mode = "kaleido".
// Number of mirror segments is baked from K_KALEIDO_SYM (set by the factory).
vec2 feedbackFn(vec2 uv, float t) {
    vec2 c = uv - 0.5;
    float r = length(c);
    float a = atan(c.y, c.x);
    float seg = EVO_TAU / float(K_KALEIDO_SYM);
    a = mod(a, seg);
    a = abs(a - seg * 0.5);
    a += 0.02 * sin(t * 0.3);
    return vec2(cos(a), sin(a)) * r + 0.5;
}
