from __future__ import annotations

from .base import ModuleSections, ModuleUniform, ShaderModule


class HeatDiffusionModule(ShaderModule):
    """Heat diffusion / thermal field."""

    def sections(self) -> ModuleSections:
        uniforms: list[ModuleUniform] = []

        speed = self._uniform("speed", 1.0, uniforms=uniforms)
        diffusion = self._uniform("diffusion", 0.1, uniforms=uniforms)

        defines = (
            f"#define K_HEAT_SPEED ({speed.name})\n#define K_HEAT_DIFFUSION ({diffusion.name})\n"
        )

        field_glsl = """
float field_heat(vec2 p, float t) {
    float n = fbm(p * 2.0 + t * K_HEAT_SPEED * 0.1, 4, 2.0, 0.5, 0.5);
    n += sin(p.x * 5.0 + t * K_HEAT_SPEED) * K_HEAT_DIFFUSION;
    n += cos(p.y * 4.0 + t * K_HEAT_SPEED * 0.7) * K_HEAT_DIFFUSION * 0.5;
    return n * 0.5 + 0.5;
}
"""

        call = "float n = field_heat(p, t);"

        return ModuleSections(
            header=defines + field_glsl,
            call=call,
            uniforms=uniforms,
        )
