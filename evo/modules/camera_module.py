from __future__ import annotations

from .base import ModuleSections, ModuleUniform, ShaderModule


class CameraModule(ShaderModule):
    """Orbit camera for SDF rendering."""

    def sections(self) -> ModuleSections:
        uniforms: list[ModuleUniform] = []

        self._uniform("camera_distance", 3.5, uniforms=uniforms)
        self._uniform("camera_orbit_speed", 1.0, uniforms=uniforms)
        cam_z = self._uniform("camera_z", 3.0, uniforms=uniforms)

        defines = f"#define K_SDF_CAMERA_Z ({cam_z.name})\n"

        # The camera functions are used within field_sdf3D, which is defined
        # in the SDF module. This module just declares the uniforms.
        return ModuleSections(header=defines, uniforms=uniforms)
