// evo/feedback/mirror.glsl
// Feedback transform: kaleidoscopic mirror fold.
vec2 feedbackFn(vec2 uv, float t) {
    vec2 c = (uv - 0.5) * 2.0;
    c = abs(c);
    c = c * 0.5 + 0.5;
    return fract(c + 0.01 * sin(t * 0.3));
}
