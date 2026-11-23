from __future__ import annotations

from .base import ModuleSections, ModuleUniform, ShaderModule


class ParticleModule(ShaderModule):
    """Organic particle field with evolved density and size."""

    def sections(self) -> ModuleSections:
        uniforms: list[ModuleUniform] = []
        density = self._uniform("particle_density", 1.0, uniforms=uniforms)
        size = self._uniform("particle_size", 0.02, uniforms=uniforms)

        defines = (
            f"#define K_PARTICLE_DENSITY ({density.name})\n#define K_PARTICLE_SIZE ({size.name})\n"
        )

        header = (
            defines
            + """
vec3 particle_field(vec2 uv, float t) {
    vec3 acc = vec3(0.0);
    for (int i = 0; i < 12; i++) {
        vec2 center = vec2(
            sin(t * 0.7 + float(i) * 1.3) * 0.6,
            cos(t * 0.5 + float(i) * 1.7) * 0.6
        );
        float d = length(uv - center);
        acc += vec3(0.02, 0.01, 0.03) * smoothstep(K_PARTICLE_SIZE * 3.0, 0.0, d);
    }
    return acc * K_PARTICLE_DENSITY;
}
"""
        )

        call = "col += particle_field(sample_uv, t);"

        return ModuleSections(header=header, call=call, uniforms=uniforms)
