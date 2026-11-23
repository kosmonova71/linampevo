// Voronoi Field Module
// Cellular/Voronoi noise field
// Genes: scale, jitter, animation_speed

#ifndef VORONOI_FIELD_GLSL
#define VORONOI_FIELD_GLSL

#include "common.glsl"

vec2 voronoi_hash(vec2 p) {
    p = vec2(dot(p, vec2(127.1, 311.7)),
             dot(p, vec2(269.5, 183.3)));
    return fract(sin(p) * 43758.5453);
}

vec3 voronoi(vec2 p, float jitter, float t) {
    vec2 n = floor(p);
    vec2 f = fract(p);
    float md = 1.0;
    float md2 = 1.0;
    vec2 mg;
    for(int y = -1; y <= 1; y++)
    for(int x = -1; x <= 1; x++) {
        vec2 g = vec2(float(x), float(y));
        vec2 o = voronoi_hash(n + g);
        o = 0.5 + 0.5 * sin(t * 0.5 + 6.2831 * o) * jitter;
        vec2 r = g + o - f;
        float d = length(r);
        if(d < md) {
            md2 = md;
            md = d;
            mg = g;
        } else if(d < md2) {
            md2 = d;
        }
    }
    return vec3(md, md2, hash21(n + mg));
}

float field_voronoi(vec2 p, float t, float noise_scale, float jitter, float animation_speed) {
    vec2 sp = p * noise_scale + t * animation_speed * 0.1;
    vec3 v = voronoi(sp, jitter, t);
    return v.x;
}

float field_voronoi_edge(vec2 p, float t, float noise_scale, float jitter, float animation_speed) {
    vec2 sp = p * noise_scale + t * animation_speed * 0.1;
    vec3 v = voronoi(sp, jitter, t);
    return v.y - v.x;
}

#endif
