from __future__ import annotations

from .base import ModuleSections, ModuleUniform, ShaderModule


class FluidFeedbackModule(ShaderModule):
    """Fluid feedback field using vortex-based domain warping."""

    def sections(self) -> ModuleSections:
        uniforms: list[ModuleUniform] = []

        strength = self._uniform("strength", 0.5, uniforms=uniforms)
        scale = self._uniform("noise_scale", 3.0, uniforms=uniforms)

        defines = (
            f"#define K_FLUID_STRENGTH ({strength.name})\n#define K_FLUID_SCALE ({scale.name})\n"
        )

        field_glsl = """
float field_fluid(vec2 p, float t) {
    vec2 uv = p;
    for (int i = 0; i < 3; i++) {
        float angle = sin(uv.y * 3.0 + t * 0.5) * K_FLUID_STRENGTH;
        float s = sin(angle);
        float c = cos(angle);
        mat2 rot = mat2(c, -s, s, c);
        uv = rot * uv * 0.9 + vec2(0.1);
        uv += vec2(sin(t * 0.3) * 0.05, cos(t * 0.4) * 0.05);
    }
    float n = fbm(uv * K_FLUID_SCALE, 4, 2.0, 0.5, 0.5);
    return n;
}
"""

        call = "float n = field_fluid(p, t);"

        return ModuleSections(
            header=defines + field_glsl,
            call=call,
            uniforms=uniforms,
        )
