from __future__ import annotations

from .base import ModuleSections, ModuleUniform, ShaderModule


class SlimeMoldModule(ShaderModule):
    """Slime mold pattern field."""

    def sections(self) -> ModuleSections:
        params = self.config.params
        uniforms: list[ModuleUniform] = []

        speed = self._uniform("speed", 1.0, uniforms=uniforms)
        scale = self._uniform("noise_scale", 3.0, uniforms=uniforms)
        iterations = int(params.get("iterations", 16))

        defines = (
            f"#define K_SLIME_ITER {iterations}\n"
            f"#define K_SLIME_SPEED ({speed.name})\n"
            f"#define K_SLIME_SCALE ({scale.name})\n"
        )

        field_glsl = """
float field_slime(vec2 p, float t) {
    vec2 uv = p * K_SLIME_SCALE;
    float n = 0.0;
    for (int i = 0; i < K_SLIME_ITER; i++) {
        float angle = sin(uv.x * 3.0 + uv.y * 2.0 + t * K_SLIME_SPEED * 0.2) * 2.0;
        float s = sin(angle);
        float c = cos(angle);
        mat2 rot = mat2(c, -s, s, c);
        uv = rot * uv * 0.95;
        n += fbm(uv + vec2(t * 0.02), 3, 2.0, 0.5, 0.5);
    }
    return n / float(K_SLIME_ITER);
}
"""

        call = "float n = field_slime(p, t);"

        return ModuleSections(
            header=defines + field_glsl,
            call=call,
            uniforms=uniforms,
        )
