from __future__ import annotations

from .base import ModuleSections, ModuleUniform, ShaderModule


class CurlWarpModule(ShaderModule):
    """Curl noise-based coordinate warp."""

    def sections(self) -> ModuleSections:
        uniforms: list[ModuleUniform] = []
        strength = self._uniform("strength", 0.5, uniforms=uniforms)
        scale = self._uniform("scale", 1.0, uniforms=uniforms)
        octaves = int(self.config.params.get("octaves", 4))

        defines = (
            f"#define K_CURL_STRENGTH ({strength.name})\n"
            f"#define K_CURL_SCALE ({scale.name})\n"
            f"#define K_CURL_OCTAVES {octaves}\n"
        )

        header = (
            defines
            + """
vec2 warp_curl(vec2 p, float t, float strength, float scale, int octaves) {
    float e = 0.01 * scale;
    float n1 = fbm(p + vec2(0.0, e), octaves, 2.0, 0.5);
    float n2 = fbm(p - vec2(0.0, e), octaves, 2.0, 0.5);
    float n3 = fbm(p + vec2(e, 0.0), octaves, 2.0, 0.5);
    float n4 = fbm(p - vec2(e, 0.0), octaves, 2.0, 0.5);
    return vec2(n1 - n2, n4 - n3) * strength;
}

vec2 apply_warp(vec2 p, float t) {
    return p + warp_curl(p * K_CURL_SCALE + t * 0.05, t, K_CURL_STRENGTH, K_CURL_SCALE, K_CURL_OCTAVES);
}
"""
        )

        return ModuleSections(header=header, uniforms=uniforms)


class VortexWarpModule(ShaderModule):
    """Vortex / spiral warp."""

    def sections(self) -> ModuleSections:
        uniforms: list[ModuleUniform] = []
        strength = self._uniform("strength", 0.3, uniforms=uniforms)
        arms = self._uniform("arms", 3.0, uniforms=uniforms)
        twist = self._uniform("twist_rate", 0.5, uniforms=uniforms)

        defines = (
            f"#define K_VORTEX_STRENGTH ({strength.name})\n"
            f"#define K_VORTEX_ARMS ({arms.name})\n"
            f"#define K_VORTEX_TWIST ({twist.name})\n"
        )

        header = (
            defines
            + """
vec2 warp_vortex(vec2 p, float t, float strength, float arms, float twist_rate) {
    float r = length(p);
    float a = atan(p.y, p.x) + strength * r + t * twist_rate;
    float arms_f = max(1.0, arms);
    a = mod(a, EVO_TAU / arms_f);
    return vec2(cos(a), sin(a)) * r;
}

vec2 apply_warp(vec2 p, float t) {
    return warp_vortex(p, t, K_VORTEX_STRENGTH, K_VORTEX_ARMS, K_VORTEX_TWIST);
}
"""
        )

        return ModuleSections(header=header, uniforms=uniforms)


class TwistWarpModule(ShaderModule):
    """Twist / shear coordinate warp."""

    def sections(self) -> ModuleSections:
        uniforms: list[ModuleUniform] = []
        amount = self._uniform("amount", 0.5, uniforms=uniforms)
        iterations = int(self.config.params.get("iterations", 2))

        defines = f"#define K_TWIST_AMOUNT ({amount.name})\n#define K_TWIST_ITER {iterations}\n"

        header = (
            defines
            + """
vec2 warp_twist(vec2 p, float t, float amount, int iterations) {
    for (int i = 0; i < 4; i++) {
        if (i >= iterations) break;
        float a = amount * length(p) + t * 0.1;
        float c = cos(a), s = sin(a);
        p = mat2(c, -s, s, c) * p;
    }
    return p;
}

vec2 apply_warp(vec2 p, float t) {
    return warp_twist(p, t, K_TWIST_AMOUNT, K_TWIST_ITER);
}
"""
        )

        return ModuleSections(header=header, uniforms=uniforms)


class NoneWarpModule(ShaderModule):
    """No-op warp: pass-through."""

    def sections(self) -> ModuleSections:
        return ModuleSections(header="vec2 apply_warp(vec2 p, float t) { return p; }\n")
