#!/usr/bin/env python3

"""Boids flocking field module.

Evaluates a procedural boids-inspired field value ``n`` inside
``main()``.  The genome's ``field_type == "boids"`` selects this
module.
"""

from __future__ import annotations

from .base import ModuleSections, ModuleUniform, ShaderModule


class BoidsModule(ShaderModule):
    """Boids flocking simulation field."""

    def sections(self) -> ModuleSections:
        uniforms: list[ModuleUniform] = []

        separation = self._uniform("separation", 1.0, uniforms=uniforms)
        alignment = self._uniform("alignment", 1.0, uniforms=uniforms)
        cohesion = self._uniform("cohesion", 1.0, uniforms=uniforms)
        visual_range = self._uniform("visual_range", 0.3, uniforms=uniforms)
        speed = self._uniform("speed", 1.0, uniforms=uniforms)
        scale = self._uniform("noise_scale", 3.0, uniforms=uniforms)

        defines = (
            f"#define K_BOIDS_SEPARATION ({separation.name})\n"
            f"#define K_BOIDS_ALIGNMENT ({alignment.name})\n"
            f"#define K_BOIDS_COHESION ({cohesion.name})\n"
            f"#define K_BOIDS_RANGE ({visual_range.name})\n"
            f"#define K_BOIDS_SPEED ({speed.name})\n"
            f"#define K_BOIDS_SCALE ({scale.name})\n"
        )

        fields_glsl = """
#ifndef FBM_5ARG_GLSL
#define FBM_5ARG_GLSL
float fbm(vec2 p, int layers, float lacunarity, float gain, float domain_warp) {
    float sum = 0.0;
    float amp = 1.0;
    float freq = 1.0;
    for (int i = 0; i < layers; i++) {
        sum += amp * sin(p.x * freq + p.y * freq + freq);
        freq *= lacunarity;
        amp *= gain;
    }
    return sum;
}
#endif

vec3 boids_separation(vec2 p, vec2 mouse, float range) {
    vec2 diff = p - mouse;
    float dist = length(diff);
    if (dist < range && dist > 0.001) {
        return normalize(diff) * (range - dist) / range;
    }
    return vec3(0.0);
}

float boids_field(vec2 p, float t, float separation, float alignment,
                  float cohesion, float range, float speed, float noise_scale) {
    vec2 mouse = vec2(sin(t * 0.3) * 0.5, cos(t * 0.4) * 0.5);
    float n_sep = length(boids_separation(p, mouse, range).xy);
    float n_align = sin(p.x * 3.0 + t * speed) * cos(p.y * 2.0 + t * speed * 0.7);
    float n_coh = length(p - mouse) / max(range, 0.001);

    vec2 offset = vec2(
        sin(n_align * separation + t * 0.5) * range * 0.5,
        cos(n_align * alignment + t * 0.3) * range * 0.5
    );

    float n = fbm(p * noise_scale + offset, 4, 2.0, 0.5, 0.5);
    n += n_sep * separation * 0.3;
    n -= n_coh * cohesion * 0.2;

    return n;
}
"""

        call = (
            "float n = boids_field(p + t * 0.1, t, K_BOIDS_SEPARATION, "
            "K_BOIDS_ALIGNMENT, K_BOIDS_COHESION, K_BOIDS_RANGE, "
            "K_BOIDS_SPEED, K_BOIDS_SCALE);"
        )

        return ModuleSections(
            header=defines + fields_glsl,
            call=call,
            uniforms=uniforms,
        )
