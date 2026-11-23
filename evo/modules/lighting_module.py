from __future__ import annotations

from .base import ModuleSections, ModuleUniform, ShaderModule


class LightingModule(ShaderModule):
    """Evolved lighting model with intensity and ambient control."""

    def sections(self) -> ModuleSections:
        uniforms: list[ModuleUniform] = []
        intensity = self._uniform("light_intensity", 0.8, uniforms=uniforms)
        ambient = self._uniform("light_ambient", 0.2, uniforms=uniforms)

        header = f"""
#define K_LIGHT_INTENSITY ({intensity.name})
#define K_LIGHT_AMBIENT ({ambient.name})

vec3 apply_lighting(vec3 base_col, vec3 normal, vec3 light_dir) {{
    float diff = max(dot(normal, light_dir), 0.0);
    return base_col * (K_LIGHT_AMBIENT + K_LIGHT_INTENSITY * diff);
}}
"""

        call = "col = apply_lighting(col, vec3(0.0, 0.0, 1.0), normalize(vec3(p, 0.5)));"

        return ModuleSections(header=header, call=call, uniforms=uniforms)
