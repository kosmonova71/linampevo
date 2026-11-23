

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

vec2 hash22(vec2 p) {
    vec3 a = fract(p.xyx * vec3(123.34, 456.21, 789.12));
    a += dot(a, a + 45.32);
    return fract(vec2(a.x * a.y, a.y * a.z));
}

float hash11(float p) {
    p = fract(p * 0.1031);
    p += dot(p, p + 33.33);
    return fract(p * (p + p));
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    float a = hash21(i);
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

// Compatibility: generic fbm function used by warp modules (e.g. evo/warp/curl.glsl).
// Signature matches evo/fields/fbm.glsl:
//   float fbm(vec2 p, int layers, float lacunarity, float gain);
// Guarded so the definition is emitted only once when common.glsl and
// evo/fields/fbm.glsl are both assembled together (avoids "fbm redefined").
#ifndef FBM_GLSL
#define FBM_GLSL
float fbm(vec2 p, int layers, float lacunarity, float gain) {
    float v = 0.0;
    float a = gain;
    mat2 m = mat2(lacunarity, lacunarity * 0.7, -lacunarity * 0.7, lacunarity);
    for (int i = 0; i < 12; i++) {
        if (i >= layers) break;
        v += a * noise(p);
        p = m * p;
        a *= gain;
    }
    return v;
}
#endif

float mod289(float x) {

    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec2 mod289(vec2 x) {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec3 mod289(vec3 x) {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 mod289(vec4 x) {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 permute(vec4 x) {
    return mod289(((x * 34.0) + 1.0) * x);
}

vec4 taylorInvSqrt(vec4 r) {
    return 1.79284291400159 - 0.85373472095314 * r;
}

#ifndef EVO_ROT2_GLSL
#define EVO_ROT2_GLSL

// Full-turn constant used by the symmetry / kaleidoscope / spiral operators.
const float EVO_TAU = 6.28318530718;

// 2D rotation matrix, shared by warp and feedback modules.
mat2 rot2(float a) {
    float c = cos(a), s = sin(a);
    return mat2(c, -s, s, c);
}

#endif
