// evo/feedback/drift.glsl
// Feedback transform: smooth directional drift (wind / current).
vec2 feedbackFn(vec2 uv, float t) {
    return uv + vec2(0.004 * sin(t * 0.2), 0.003 * cos(t * 0.17));
}
