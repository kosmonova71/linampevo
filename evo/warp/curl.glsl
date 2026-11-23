// Curl Warp Module
// Curl noise-based coordinate transformation
// Genes: strength, scale, octaves

#ifndef CURL_WARP_GLSL
#define CURL_WARP_GLSL

#include "common.glsl"

vec2 curl_noise(vec2 p, float scale, int octaves) {
    float e = 0.01 * scale;
    float n1 = fbm(p + vec2(0.0, e), octaves, 2.0, 0.5);
    float n2 = fbm(p - vec2(0.0, e), octaves, 2.0, 0.5);
    float n3 = fbm(p + vec2(e, 0.0), octaves, 2.0, 0.5);
    float n4 = fbm(p - vec2(e, 0.0), octaves, 2.0, 0.5);
    return vec2(n1 - n2, n4 - n3);
}

vec2 warp_curl(vec2 p, float t, float strength, float scale, int octaves) {
    return curl_noise(p * scale + t * 0.05, scale, octaves) * strength;
}

#endif
