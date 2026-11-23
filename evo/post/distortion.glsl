// Distortion Post-Processing Module
// Screen space distortion effects
// Genes: amount, mode

#ifndef DISTORTION_POST_GLSL
#define DISTORTION_POST_GLSL

vec2 distortion(vec2 uv, float t, float amount, int mode) {
    vec2 result = uv;
    if(mode == 0) {
        result = uv;
    } else if(mode == 1) {
        result = uv + vec2(sin(uv.y * 10.0 + t), cos(uv.x * 10.0 + t)) * amount * 0.01;
    } else if(mode == 2) {
        float d = length(uv - 0.5);
        result = uv + normalize(uv - 0.5) * sin(d * 20.0 - t * 2.0) * amount * 0.005;
    } else if(mode == 3) {
        result = uv * (1.0 + amount * 0.1 * sin(t * 0.5));
    } else if(mode == 4) {
        float a = atan(uv.y - 0.5, uv.x - 0.5);
        float r = length(uv - 0.5);
        result = 0.5 + vec2(cos(a + amount * 0.1), sin(a + amount * 0.1)) * r;
    }
    return result;
}

#endif
