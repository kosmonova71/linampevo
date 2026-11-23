// evo/feedback/zoom.glsl
// Feedback transform: pulsing zoom (breathing lens).
vec2 feedbackFn(vec2 uv, float t) {
    vec2 c = uv - 0.5;
    c *= 1.0 - 0.012 * sin(t * 0.6);
    return c + 0.5;
}
