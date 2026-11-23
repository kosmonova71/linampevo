from __future__ import annotations

from .base import ModuleSections, ModuleUniform, ShaderModule


class PhysarumModule(ShaderModule):
    """Physarum slime mold transport network field."""

    def sections(self) -> ModuleSections:
        uniforms: list[ModuleUniform] = []

        speed = self._uniform("speed", 1.0, uniforms=uniforms)
        scale = self._uniform("noise_scale", 3.0, uniforms=uniforms)

        defines = f"#define K_PHYS_SPEED ({speed.name})\n#define K_PHYS_SCALE ({scale.name})\n"

        field_glsl = """
float field_physarum(vec2 p, float t) {
    vec2 uv = p * K_PHYS_SCALE;
    float n = 0.0;
    for (int i = 0; i < 3; i++) {
        float angle = sin(uv.y * 5.0 + t * K_PHYS_SPEED * 0.3) * 1.5;
        float s = sin(angle);
        float c = cos(angle);
        mat2 rot = mat2(c, -s, s, c);
        uv = rot * uv;
        n += fbm(uv + t * 0.05, 3, 2.0, 0.5, 0.5);
        uv *= 1.5;
    }
    return n / 3.0;
}
"""

        call = "float n = field_physarum(p, t);"

        return ModuleSections(
            header=defines + field_glsl,
            call=call,
            uniforms=uniforms,
        )
