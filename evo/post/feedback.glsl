// Feedback Post-Processing Module
// Feedback buffer with various modes
// Genes: mode, speed, amount

#ifndef FEEDBACK_POST_GLSL
#define FEEDBACK_POST_GLSL

vec2 feedback_transform(vec2 uv, float t, float speed, int mode, float amount) {
    vec2 result = uv;
    if(mode == 0) {
        result = uv;
    } else if(mode == 1) {
        float angle = atan(uv.y - 0.5, uv.x - 0.5) + 0.1 * sin(t * speed * 0.3);
        float dist = length(uv - 0.5);
        result = 0.5 + vec2(cos(angle), sin(angle)) * dist * (1.0 + amount * 0.1);
    } else if(mode == 2) {
        float dist = length(uv - 0.5);
        result = 0.5 + (uv - 0.5) * (1.0 + amount * 0.02 * sin(t * speed * 0.5));
    } else if(mode == 3) {
        result = abs(fract(uv * 2.0 + 0.5 * sin(t * speed * 0.2)) - 0.5);
    } else if(mode == 4) {
        result = fract(uv + amount * 0.02 * sin(t * speed * 0.4));
    } else if(mode == 5) {
        result = uv * (1.0 - amount * 0.005 * sin(t * speed * 0.3));
    } else if(mode == 6) {
        float a = t * speed * 0.5;
        result = vec2(uv.x * cos(a) - uv.y * sin(a), uv.x * sin(a) + uv.y * cos(a)) * (1.0 + amount * 0.01);
    }
    return result;
}

#endif
