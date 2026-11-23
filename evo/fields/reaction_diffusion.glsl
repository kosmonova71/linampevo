// Reaction Diffusion Field Module
// Gray-Scott reaction diffusion simulation
// Genes: feed_rate, kill_rate, diffusion_u, diffusion_v

#ifndef REACTION_DIFFUSION_GLSL
#define REACTION_DIFFUSION_GLSL

#include "common.glsl"

vec2 reaction_diffusion(vec2 uv, float t, float feed_rate, float kill_rate,
                        float diff_u, float diff_v, int iterations) {
    float u = 0.5;
    float v = 0.25;
    float dt = 1.0;
    float dx = 1.0;

    // Initial perturbation
    vec2 center = vec2(0.5 + 0.3 * sin(t * 0.2), 0.5 + 0.3 * cos(t * 0.3));
    if(length(uv - center) < 0.05) {
        v = 0.9;
        u = 0.1;
    }

    // Second perturbation
    center = vec2(0.5 + 0.4 * cos(t * 0.15), 0.5 + 0.4 * sin(t * 0.25));
    if(length(uv - center) < 0.03) {
        v = 0.95;
        u = 0.05;
    }

    for(int i = 0; i < 64; i++) {
        if(i >= iterations) break;

        float lap_u = 0.0;
        float lap_v = 0.0;

        vec2 offsets[4];
        offsets[0] = vec2(1.0, 0.0);
        offsets[1] = vec2(-1.0, 0.0);
        offsets[2] = vec2(0.0, 1.0);
        offsets[3] = vec2(0.0, -1.0);

        for(int j = 0; j < 4; j++) {
            vec2 neighbor = uv + offsets[j] * dx;
            float nu = 0.5 + 0.2 * noise(neighbor * 3.0 + t * 0.1);
            float nv = 0.25 + 0.1 * noise(neighbor * 4.0 + t * 0.15);
            lap_u += nu - u;
            lap_v += nv - v;
        }

        float uvv = u * v * v;
        float du = diff_u * lap_u - uvv + feed_rate * (1.0 - u);
        float dv = diff_v * lap_v + uvv - (kill_rate + feed_rate) * v;

        u += du * dt;
        v += dv * dt;
        u = clamp(u, 0.0, 1.0);
        v = clamp(v, 0.0, 1.0);
    }

    return vec2(u, v);
}

float field_rd(vec2 p, float t, float feed_rate, float kill_rate,
               float diff_u, float diff_v, int iterations) {
    vec2 rv = reaction_diffusion(p, t, feed_rate, kill_rate, diff_u, diff_v, iterations);
    return rv.y;
}

#endif
