from __future__ import annotations

from .base import ModuleSections, ModuleUniform, ShaderModule


class ErosionModule(ShaderModule):
    """Hydraulic erosion-inspired field."""

    def sections(self) -> ModuleSections:
        params = self.config.params
        uniforms: list[ModuleUniform] = []

        strength = self._uniform("strength", 0.5, uniforms=uniforms)
        iterations = int(params.get("iterations", 16))

        defines = (
            f"#define K_EROSION_ITER {iterations}\n#define K_EROSION_STRENGTH ({strength.name})\n"
        )

        field_glsl = """
float field_erosion(vec2 p, float t) {
    float n = 0.0;
    float base = fbm(p * 2.0, 4, 2.0, 0.5, 0.5);
    for (int i = 0; i < K_EROSION_ITER; i++) {
        float deposit = smoothstep(0.2, 0.8, base + n);
        n += deposit * K_EROSION_STRENGTH * 0.01;
        n -= (1.0 - deposit) * K_EROSION_STRENGTH * 0.005;
    }
    return n + base * 0.5;
}
"""

        call = "float n = field_erosion(p, t);"

        return ModuleSections(
            header=defines + field_glsl,
            call=call,
            uniforms=uniforms,
        )
