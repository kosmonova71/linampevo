from __future__ import annotations

from .base import ModuleSections, ModuleUniform, ShaderModule


class VolumeModule(ShaderModule):
    """Fog / atmospheric scattering for depth-based scenes."""

    def sections(self) -> ModuleSections:
        uniforms: list[ModuleUniform] = []
        fog_density = self._uniform("fog_density", 1.0, uniforms=uniforms)
        fog_color_r = self._uniform("fog_color_r", 0.03, uniforms=uniforms)
        fog_color_g = self._uniform("fog_color_g", 0.01, uniforms=uniforms)
        fog_color_b = self._uniform("fog_color_b", 0.06, uniforms=uniforms)

        header = f"""
#define K_FOG_DENSITY ({fog_density.name})
#define K_FOG_COLOUR vec3({fog_color_r.name}, {fog_color_g.name}, {fog_color_b.name})

vec3 apply_fog(vec3 col, float depth) {{
    float fog = 1.0 - exp(-depth * K_FOG_DENSITY * 0.08);
    return mix(col, K_FOG_COLOUR, fog);
}}
"""

        call = "col = apply_fog(col, length(p) * 0.5);"

        return ModuleSections(header=header, call=call, uniforms=uniforms)
