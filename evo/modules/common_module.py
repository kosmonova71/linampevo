from __future__ import annotations

from .base import ModuleConfig, ModuleSections, ShaderModule


class CommonModule(ShaderModule):
    """The common module provides shared GLSL boilerplate
    (precision, standard uniforms, utility functions) that
    every generated shader must include.
    """

    def __init__(self, config: ModuleConfig | None = None) -> None:
        super().__init__(config or ModuleConfig("common", "common", {}))

    def sections(self) -> ModuleSections:
        return ModuleSections(
            header=(
                "precision highp float;\n"
                "precision highp int;\n"
                "\n"
                "const float EVO_TAU = 6.28318530718;\n"
                "\n"
                "float hash21(vec2 p) {\n"
                "    p = fract(p * vec2(123.34, 456.21));\n"
                "    p += dot(p, p + 45.32);\n"
                "    return fract(p.x * p.y);\n"
                "}\n"
                "\n"
                "vec2 hash22(vec2 p) {\n"
                "    float n = sin(dot(p, vec2(41.0, 289.0)));\n"
                "    return fract(vec2(262144.0, 32768.0) * n);\n"
                "}\n"
                "\n"
                "float vnoise(vec2 p) {\n"
                "    vec2 i = floor(p);\n"
                "    vec2 f = fract(p);\n"
                "    float a = hash21(i);\n"
                "    float b = hash21(i + vec2(1.0, 0.0));\n"
                "    float c = hash21(i + vec2(0.0, 1.0));\n"
                "    float d = hash21(i + vec2(1.0, 1.0));\n"
                "    vec2 u = f * f * (3.0 - 2.0 * f);\n"
                "    return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;\n"
                "}\n"
                "\n"
                "float noise(vec2 p) {\n"
                "    return vnoise(p);\n"
                "}\n"
                "\n"
                "#ifndef FBM_GLSL\n"
                "#define FBM_GLSL\n"
                "float fbm(vec2 p, int layers, float lacunarity, float gain) {\n"
                "    float v = 0.0;\n"
                "    float a = gain;\n"
                "    mat2 m = mat2(lacunarity, lacunarity * 0.7, -lacunarity * 0.7, lacunarity);\n"
                "    for (int i = 0; i < 12; i++) {\n"
                "        if (i >= layers) break;\n"
                "        v += a * noise(p);\n"
                "        p = m * p;\n"
                "        a *= gain;\n"
                "    }\n"
                "    return v;\n"
                "}\n"
                "#endif\n"
                "\n"
                "#ifndef FBM_5ARG_GLSL\n"
                "#define FBM_5ARG_GLSL\n"
                "float fbm(vec2 p, int layers, float lacunarity, float gain, float domain_warp) {\n"
                "    vec2 wp = p;\n"
                "    if(domain_warp > 0.01) {\n"
                "        wp += vec2(fbm(p + 0.1, layers, lacunarity, gain),\n"
                "                   fbm(p + 5.0, layers, lacunarity, gain)) * domain_warp * 0.3;\n"
                "    }\n"
                "    return fbm(wp, layers, lacunarity, gain);\n"
                "}\n"
                "#endif\n"
                "\n"
                "#ifndef ROT2_GLSL\n"
                "#define ROT2_GLSL\n"
                "mat2 rot2(float a) {\n"
                "    float c = cos(a), s = sin(a);\n"
                "    return mat2(c, -s, s, c);\n"
                "}\n"
                "#endif\n"
            ),
            call=("// Common module — shared shader boilerplate\n"),
        )

    def mutate(self, strength: float = 0.1) -> None:
        pass
