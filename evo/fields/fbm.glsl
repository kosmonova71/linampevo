// FBM Field Module
// Fractal Brownian Motion noise field
// Genes: fractal_layers, lacunarity, gain, warp_amount

#ifndef FBM_FIELD_GLSL
#define FBM_FIELD_GLSL

#include "common.glsl"

#ifndef FBM_GLSL
#define FBM_GLSL
float fbm(vec2 p, int layers, float lacunarity, float gain) {
    float v = 0.0;
    float a = gain;
    mat2 m = mat2(lacunarity, lacunarity * 0.7, -lacunarity * 0.7, lacunarity);
    for(int i = 0; i < 12; i++) {
        if(i >= layers) break;
        v += a * noise(p);
        p = m * p;
        a *= gain;
    }
    return v;
}
#endif

float field_fbm(vec2 p, float t, float noise_scale, int fractal_layers,
                float lacunarity, float gain, float domain_warp) {
    vec2 wp = p;
    if(domain_warp > 0.01) {
        wp += vec2(fbm(p + t * 0.1, fractal_layers, lacunarity, gain),
                   fbm(p + t * 0.1 + 5.0, fractal_layers, lacunarity, gain)) * domain_warp * 0.3;
    }
    return fbm(wp * noise_scale, fractal_layers, lacunarity, gain);
}

#endif
