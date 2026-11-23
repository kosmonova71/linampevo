// evo/feedback/spiral.glsl
// Feedback transform: rotate + breathe around the centre.
vec2 feedbackFn(vec2 uv, float t) {
    vec2 c = uv - 0.5;
    float a = 0.05 * sin(t * 0.3) + 0.02;
    c = rot2(a) * c;
    c *= 1.0 + 0.01 * sin(t * 0.5);
    return c + 0.5;
}
