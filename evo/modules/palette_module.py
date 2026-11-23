from __future__ import annotations

from .base import ModuleSections, ModuleUniform, ShaderModule


class CosinePaletteModule(ShaderModule):
    """Inigo Quilez cosine palette."""

    def sections(self) -> ModuleSections:
        uniforms: list[ModuleUniform] = []
        pid = int(self.config.params.get("palette_id", 0))
        shift = self._uniform("colour_shift", 0.0, uniforms=uniforms)
        speed = self._uniform("shift_speed", 0.1, uniforms=uniforms)

        defines = (
            f"#define K_PAL_ID {pid}\n"
            f"#define K_PAL_SHIFT ({shift.name})\n"
            f"#define K_PAL_SPEED ({speed.name})\n"
        )

        header = (
            defines
            + """
vec3 palette_cosine(float t, int palette_id, float colour_shift, float t_mod) {
    float shift = t_mod * colour_shift;
    vec3 a, b, c, d;

    if (palette_id == 0) {
        a = vec3(0.15, 0.20, 0.35);
        b = vec3(0.65, 0.25, 0.75);
        c = vec3(1.00, 0.95, 0.85);
        d = vec3(0.20, 0.40, 0.60);
    } else if (palette_id == 1) {
        a = vec3(0.05, 0.10, 0.20);
        b = vec3(0.90, 0.40, 0.20);
        c = vec3(2.00, 1.80, 1.60);
        d = vec3(0.00, 0.33, 0.67);
    } else if (palette_id == 2) {
        a = vec3(0.50, 0.50, 0.50);
        b = vec3(0.50, 0.50, 0.50);
        c = vec3(1.00, 1.00, 1.00);
        d = vec3(0.00, 0.33, 0.67);
    } else if (palette_id == 3) {
        a = vec3(0.80, 0.20, 0.20);
        b = vec3(0.20, 0.60, 0.80);
        c = vec3(2.00, 1.00, 1.50);
        d = vec3(0.00, 0.10, 0.20);
    } else if (palette_id == 4) {
        a = vec3(0.10, 0.30, 0.10);
        b = vec3(0.70, 0.90, 0.30);
        c = vec3(1.50, 2.00, 1.00);
        d = vec3(0.00, 0.50, 0.00);
    } else {
        a = vec3(0.20, 0.00, 0.30);
        b = vec3(0.80, 0.60, 0.90);
        c = vec3(1.00, 1.50, 2.00);
        d = vec3(0.30, 0.00, 0.50);
    }

    return a + b * cos(6.28318 * (c * t + d) + vec3(shift, shift + 2.094, shift + 4.189));
}
"""
        )

        call = "vec3 _pal_col = palette_cosine(hueT, K_PAL_ID, K_PAL_SHIFT, t * K_PAL_SPEED);"

        return ModuleSections(header=header, call=call, uniforms=uniforms)


class SpectralPaletteModule(ShaderModule):
    """Wavelength-based spectral palette."""

    def sections(self) -> ModuleSections:
        uniforms: list[ModuleUniform] = []
        wl = self._uniform("wavelength_shift", 0.0, uniforms=uniforms)
        disp = self._uniform("dispersion", 0.0, uniforms=uniforms)

        defines = f"#define K_SPEC_WL_SHIFT ({wl.name})\n#define K_SPEC_DISPERSION ({disp.name})\n"

        header = (
            defines
            + """
vec3 palette_spectral(float t, float wl_shift, float dispersion) {
    float wl = 380.0 + t * 400.0 + wl_shift * 100.0 + dispersion * 100.0 * sin(t * 5.0);
    vec3 c;
    if (wl < 440.0) {
        c = vec3(-(wl - 440.0) / 60.0, 0.0, 1.0);
    } else if (wl < 490.0) {
        c = vec3(0.0, (wl - 440.0) / 50.0, 1.0);
    } else if (wl < 510.0) {
        c = vec3(0.0, 1.0, -(wl - 510.0) / 20.0);
    } else if (wl < 580.0) {
        c = vec3((wl - 510.0) / 70.0, 1.0, 0.0);
    } else if (wl < 645.0) {
        c = vec3(1.0, -(wl - 645.0) / 65.0, 0.0);
    } else {
        c = vec3(1.0, 0.0, 0.0);
    }
    float fade = smoothstep(380.0, 420.0, wl) * smoothstep(780.0, 720.0, wl);
    return c * fade;
}
"""
        )

        call = "vec3 _pal_col = palette_spectral(hueT, K_SPEC_WL_SHIFT, K_SPEC_DISPERSION);"

        return ModuleSections(header=header, call=call, uniforms=uniforms)


class PlasmaPaletteModule(ShaderModule):
    """Plasma / organic palette."""

    def sections(self) -> ModuleSections:
        uniforms: list[ModuleUniform] = []
        speed = self._uniform("speed", 1.0, uniforms=uniforms)
        complexity = self._uniform("complexity", 1.0, uniforms=uniforms)
        sat = self._uniform("saturation", 1.0, uniforms=uniforms)

        defines = (
            f"#define K_PLASMA_SPEED ({speed.name})\n"
            f"#define K_PLASMA_COMPLEXITY ({complexity.name})\n"
            f"#define K_PLASMA_SAT ({sat.name})\n"
        )

        header = (
            defines
            + """
vec3 palette_plasma(float t, float speed, float complexity, float saturation) {
    vec3 col = vec3(
        sin(t * complexity * 6.0 + speed * 0.5),
        sin(t * complexity * 6.0 + 2.094 + speed * 0.7),
        sin(t * complexity * 6.0 + 4.189 + speed * 0.3)
    );
    col = col * 0.5 + 0.5;
    col = mix(vec3(0.5), col, saturation);
    return col;
}
"""
        )

        call = "vec3 _pal_col = palette_plasma(hueT, K_PLASMA_SPEED, K_PLASMA_COMPLEXITY, K_PLASMA_SAT);"

        return ModuleSections(header=header, call=call, uniforms=uniforms)
