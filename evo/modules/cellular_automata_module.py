from __future__ import annotations

from .base import ModuleSections, ModuleUniform, ShaderModule


class CellularAutomataModule(ShaderModule):
    """Cellular automata field."""

    def sections(self) -> ModuleSections:
        params = self.config.params
        uniforms: list[ModuleUniform] = []

        rule = int(params.get("rule", 30))
        speed = self._uniform("speed", 1.0, uniforms=uniforms)

        defines = f"#define K_CA_RULE {rule}\n#define K_CA_SPEED ({speed.name})\n"

        field_glsl = """
float field_ca(vec2 p, float t) {
    float cell_step = 1.0 / 512.0;
    float x = p.x;
    float y = p.y;
    int ix = int(floor(x / cell_step));
    int iy = int(floor(y / cell_step));
    int n = (ix + iy * 512) % 256;
    float state = float(n) / 255.0;
    for (int i = 0; i < 8; i++) {
        int neighbor = (ix + iy * 512 + i * 7 + 1) % 256;
        state += float(neighbor) / 255.0 * 0.125;
    }
    state = sin(state * 3.14159 + t * K_CA_SPEED);
    return state * 0.5 + 0.5;
}
"""

        call = "float n = field_ca(p, t);"

        return ModuleSections(
            header=defines + field_glsl,
            call=call,
            uniforms=uniforms,
        )
