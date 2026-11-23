from __future__ import annotations

from .base import ModuleSections, ModuleUniform, ShaderModule


class WaveEquationModule(ShaderModule):
    """Wave equation simulation field."""

    def sections(self) -> ModuleSections:
        uniforms: list[ModuleUniform] = []

        frequency = self._uniform("frequency", 3.0, uniforms=uniforms)
        amplitude = self._uniform("amplitude", 1.0, uniforms=uniforms)
        damping = self._uniform("damping", 0.5, uniforms=uniforms)

        defines = (
            f"#define K_WAVE_FREQ ({frequency.name})\n"
            f"#define K_WAVE_AMP ({amplitude.name})\n"
            f"#define K_WAVE_DAMP ({damping.name})\n"
        )

        field_glsl = """
float field_wave(vec2 p, float t) {
    float d = length(p);
    float wave = sin(d * K_WAVE_FREQ - t * K_WAVE_FREQ * K_WAVE_AMP) * exp(-d * K_WAVE_DAMP);
    wave += sin(d * K_WAVE_FREQ * 1.5 + t * K_WAVE_FREQ * 0.7) * exp(-d * K_WAVE_DAMP * 0.8) * 0.5;
    return wave * 0.5 + 0.5;
}
"""

        call = "float n = field_wave(p, t);"

        return ModuleSections(
            header=defines + field_glsl,
            call=call,
            uniforms=uniforms,
        )
