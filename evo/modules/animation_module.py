#!/usr/bin/env python3
"""animation_module.py

Milkdrop-style per-frame animation module.
Provides animated parameters (zoom, rot, cx, cy, warp, waves) that update
smoothly each frame based on audio and time.
"""

from __future__ import annotations

from evo.modules.base import ModuleConfig, ModuleSections, ModuleUniform, ShaderModule


class AnimationModule(ShaderModule):
    """Per-frame animation module providing Milkdrop-style animated parameters."""

    CATEGORY = "animation"

    def __init__(self, config: ModuleConfig):
        super().__init__(config)
        # Animation parameters are passed via genome.animation dict
        self.params = config.params or {}

    def sections(self) -> ModuleSections:
        secs = ModuleSections()

        # Uniform declarations for animation parameters (no prefix to match shader template)
        anim_uniforms = [
            "anim_zoom", "anim_rot", "anim_cx", "anim_cy",
            "anim_dx", "anim_dy", "anim_sx", "anim_sy",
            "anim_warp", "anim_wave_r", "anim_wave_g", "anim_wave_b",
            "anim_wave_x", "anim_wave_y",
            "anim_ob_size", "anim_ob_r", "anim_ob_g", "anim_ob_b", "anim_ob_a",
            "anim_ib_size", "anim_ib_r", "anim_ib_g", "anim_ib_b", "anim_ib_a",
            "anim_mv_x", "anim_mv_y", "anim_mv_l", "anim_mv_r",
            "anim_mv_g", "anim_mv_b", "anim_mv_a",
        ]
        for name in anim_uniforms:
            secs.uniforms.append(
                ModuleUniform(name=name, value=float(self.params.get(name, 0.0)), kind="float")
            )

        # Animation helper functions
        secs.header = """
// Milkdrop-style per-frame animation
float anim_progress(float t, float speed) {
    return fract(t * speed);
}

vec2 apply_anim_zoom(vec2 uv, float zoom, float cx, float cy) {
    vec2 centered = uv - vec2(cx, cy);
    return centered * zoom + vec2(cx, cy);
}

vec2 apply_anim_rot(vec2 uv, float rot, float cx, float cy) {
    vec2 centered = uv - vec2(cx, cy);
    float c = cos(rot);
    float s = sin(rot);
    return vec2(centered.x * c - centered.y * s,
                centered.x * s + centered.y * c) + vec2(cx, cy);
}

vec2 apply_anim_warp(vec2 uv, float warp) {
    // Milkdrop-style warp: radial distortion
    float r = length(uv - 0.5);
    float theta = atan(uv.y - 0.5, uv.x - 0.5);
    r += warp * sin(r * 20.0 + theta * 3.0);
    return vec2(0.5) + r * vec2(cos(theta), sin(theta));
}

vec2 apply_anim_transform(vec2 uv) {
    // Apply full Milkdrop transform chain
    uv = apply_anim_zoom(uv, anim_zoom, anim_cx, anim_cy);
    uv = apply_anim_rot(uv, anim_rot, anim_cx, anim_cy);
    uv += vec2(anim_dx, anim_dy);
    uv = (uv - 0.5) * vec2(anim_sx, anim_sy) + 0.5;
    if (anim_warp > 0.001) {
        uv = apply_anim_warp(uv, anim_warp);
    }
    return uv;
}

// Wave functions for color/shape modulation
float wave_r(float t) { return anim_wave_r + 0.3 * sin(t * 0.7); }
float wave_g(float t) { return anim_wave_g + 0.3 * sin(t * 0.7 + 2.094); }
float wave_b(float t) { return anim_wave_b + 0.3 * sin(t * 0.7 + 4.189); }
float wave_x(float t) { return anim_wave_x + 0.2 * sin(t * 0.5); }
float wave_y(float t) { return anim_wave_y + 0.2 * cos(t * 0.5); }

// Outer border
vec4 outer_border(vec2 uv) {
    float d = length(uv - 0.5) * 2.0;
    float border = smoothstep(1.0 - anim_ob_size, 1.0, d);
    return vec4(anim_ob_r, anim_ob_g, anim_ob_b, anim_ob_a) * border;
}

// Inner border
vec4 inner_border(vec2 uv) {
    float d = length(uv - 0.5) * 2.0;
    float border = smoothstep(anim_ib_size, anim_ib_size + 0.05, d);
    return vec4(anim_ib_r, anim_ib_g, anim_ib_b, anim_ib_a) * border;
}

// Motion vectors
vec2 motion_vector(vec2 uv) {
    return vec2(anim_mv_x, anim_mv_y) * length(uv - 0.5) * 2.0;
}
"""
        return secs


def register(registry):
    registry.register("animation", "default", AnimationModule)