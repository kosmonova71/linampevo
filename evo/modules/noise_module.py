from __future__ import annotations

from pathlib import Path

from .base import ModuleSections, ModuleUniform, ShaderModule


class FBMFieldModule(ShaderModule):
    """Fractal Brownian Motion noise field."""

    def sections(self) -> ModuleSections:
        params = self.config.params
        uniforms: list[ModuleUniform] = []

        layers = int(params.get("fractal_layers", 4))
        if self.config.fractal_layers is not None:
            layers = min(layers, int(self.config.fractal_layers))

        lac = self._uniform("lacunarity", 2.0, uniforms=uniforms)
        gain = self._uniform("gain", 0.5, uniforms=uniforms)
        dw = self._uniform("domain_warp", 0.5, uniforms=uniforms)
        ns = self._uniform("noise_scale", 3.0, uniforms=uniforms)

        defines = (
            f"#define K_FRACTAL_LAYERS {layers}\n"
            f"#define K_LACUNARITY ({lac.name})\n"
            f"#define K_GAIN ({gain.name})\n"
            f"#define K_DOMAIN_WARP ({dw.name})\n"
            f"#define K_NOISE_SCALE ({ns.name})\n"
        )

        fields_glsl = """
float field_fbm(vec2 p, float t, float noise_scale, int fractal_layers,
                float lacunarity, float gain, float domain_warp) {{
    vec2 wp = p;
    if(domain_warp > 0.01) {{
        wp += vec2(fbm(p + t * 0.1, fractal_layers, lacunarity, gain),
                   fbm(p + t * 0.1 + 5.0, fractal_layers, lacunarity, gain)) * domain_warp * 0.3;
    }}
    return fbm(wp * noise_scale, fractal_layers, lacunarity, gain);
}}
"""

        call = (
            "float n = field_fbm(p + t * 0.05, t, K_NOISE_SCALE, "
            "K_FRACTAL_LAYERS, K_LACUNARITY, K_GAIN, K_DOMAIN_WARP);"
        )

        return ModuleSections(
            header=defines + fields_glsl,
            call=call,
            uniforms=uniforms,
        )


class VoronoiFieldModule(ShaderModule):
    """Voronoi noise field."""

    def sections(self) -> ModuleSections:
        uniforms: list[ModuleUniform] = []
        ns = self._uniform("noise_scale", 5.0, uniforms=uniforms)
        jit = self._uniform("jitter", 1.0, uniforms=uniforms)
        spd = self._uniform("animation_speed", 1.0, uniforms=uniforms)

        defines = (
            f"#define K_VORONOI_SCALE ({ns.name})\n"
            f"#define K_VORONOI_JITTER ({jit.name})\n"
            f"#define K_VORONOI_SPEED ({spd.name})\n"
        )

        field_glsl = """
float field_voronoi(vec2 p, float t, float scale, float jitter, float speed) {
    vec2 n = floor(p * scale);
    vec2 f = fract(p * scale);
    float md = 8.0;
    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            vec2 g = vec2(float(i), float(j));
            vec2 o = 0.5 + 0.5 * sin(t * speed + 6.2831 * hash22(n + g));
            o = -0.5 + jitter * (o - 0.5);
            vec2 d = g + o - f;
            float dist = dot(d, d);
            if (dist < md) md = dist;
        }
    }
    return sqrt(md);
}
"""

        call = "float n = field_voronoi(p, t, K_VORONOI_SCALE, K_VORONOI_JITTER, K_VORONOI_SPEED);"

        return ModuleSections(
            header=defines + field_glsl,
            call=call,
            uniforms=uniforms,
        )


class ReactionDiffusionModule(ShaderModule):
    """Reaction-diffusion pattern field."""

    def sections(self) -> ModuleSections:
        uniforms: list[ModuleUniform] = []
        feed = self._uniform("feed_rate", 0.055, uniforms=uniforms)
        kill = self._uniform("kill_rate", 0.062, uniforms=uniforms)
        du = self._uniform("diffusion_u", 1.0, uniforms=uniforms)
        dv = self._uniform("diffusion_v", 0.5, uniforms=uniforms)
        iterations = int(self.config.params.get("iterations", 32))

        defines = (
            f"#define K_RD_FEED ({feed.name})\n"
            f"#define K_RD_KILL ({kill.name})\n"
            f"#define K_RD_DIFF_U ({du.name})\n"
            f"#define K_RD_DIFF_V ({dv.name})\n"
            f"#define K_RD_ITERATIONS {iterations}\n"
        )

        field_glsl = """
float field_rd(vec2 uv, float t, float feed, float kill,
               float diffU, float diffV, int iterations) {
    float u = 0.5 + 0.1 * sin(uv.x * 3.0 + t);
    float v = 0.25 + 0.05 * cos(uv.y * 4.0 + t * 0.8);
    float dt = 0.1;
    vec2 p = uv;
    for (int i = 0; i < 64; i++) {
        if (i >= iterations) break;
        vec2 n = vec2(
            fract(sin(dot(floor(p * 4.0), vec2(12.9898, 78.233))) * 43758.5453),
            fract(sin(dot(floor(p * 4.0 + 100.0), vec2(12.9898, 78.233))) * 43758.5453)
        );
        float lap = n.x - u;
        float duv = diffU * lap - feed * kill * u * v + float(i) * 0.001;
        float dvv = diffV * lap + feed * kill * u * v - (kill + feed) * v + float(i) * 0.001;
        u = clamp(u + duv * dt, 0.0, 1.0);
        v = clamp(v + dvv * dt, 0.0, 1.0);
        p += vec2(0.01);
    }
    return u;
}
"""

        call = "float n = field_rd(uv, t, K_RD_FEED, K_RD_KILL, K_RD_DIFF_U, K_RD_DIFF_V, K_RD_ITERATIONS);"

        return ModuleSections(
            header=defines + field_glsl,
            call=call,
            uniforms=uniforms,
        )


class MandelbrotFieldModule(ShaderModule):
    """Mandelbrot / Julia fractal field."""

    def sections(self) -> ModuleSections:
        uniforms: list[ModuleUniform] = []
        max_iter = int(self.config.params.get("max_iter", 64))
        if self.config.raymarch_steps is not None:
            max_iter = min(max_iter, int(self.config.raymarch_steps))

        esc = self._uniform("escape_radius", 4.0, uniforms=uniforms)
        zoom = self._uniform("zoom", 1.0, uniforms=uniforms)
        julia = int(self.config.params.get("julia_mode", 0))
        jcx = self._uniform("julia_cx", 0.355, uniforms=uniforms)
        jcy = self._uniform("julia_cy", 0.355, uniforms=uniforms)

        defines = (
            f"#define K_MB_MAX_ITER {max_iter}\n"
            f"#define K_MB_ESCAPE ({esc.name})\n"
            f"#define K_MB_ZOOM ({zoom.name})\n"
            f"#define K_MB_JULIA {julia}\n"
            f"#define K_MB_JULIA_CX ({jcx.name})\n"
            f"#define K_MB_JULIA_CY ({jcy.name})\n"
        )

        field_glsl = """
float field_mandelbrot(vec2 p, float t, int max_iter, float escape,
                       float zoom, int julia, float jcx, float jcy) {
    vec2 z = julia != 0 ? p : vec2(0.0);
    vec2 c = julia != 0 ? vec2(jcx, jcy) : p * zoom + vec2(sin(t * 0.05) * 0.1, cos(t * 0.03) * 0.1);
    float it = 0.0;
    for (int i = 0; i < 256; i++) {
        if (i >= max_iter) break;
        z = vec2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
        if (dot(z, z) > escape) break;
        it += 1.0;
    }
    return it / float(max_iter);
}
"""

        call = "float n = field_mandelbrot(p, t, K_MB_MAX_ITER, K_MB_ESCAPE, K_MB_ZOOM, K_MB_JULIA, K_MB_JULIA_CX, K_MB_JULIA_CY);"

        return ModuleSections(
            header=defines + field_glsl,
            call=call,
            uniforms=uniforms,
        )


class SDFFieldModule(ShaderModule):
    """Signed distance field (3D raymarched)."""

    def sections(self) -> ModuleSections:
        uniforms: list[ModuleUniform] = []

        prim = int(self.config.params.get("primitive_type", 0))
        shape = int(self.config.params.get("sdf_shape", 0))
        shape = max(0, min(9, shape))

        roundness = self._uniform("sdf_roundness", 0.5, uniforms=uniforms)
        surf_noise = self._uniform("surface_noise", 0.8, uniforms=uniforms)
        cam_dist = self._uniform("camera_distance", 3.5, uniforms=uniforms)
        cam_orbit = self._uniform("camera_orbit_speed", 1.0, uniforms=uniforms)
        light_rot = self._uniform("light_rotation", 1.0, uniforms=uniforms)
        curl = self._uniform("sdf_curl", 0.5, uniforms=uniforms)
        turb = self._uniform("sdf_turbulence", 0.5, uniforms=uniforms)
        dom_warp = self._uniform("sdf_dom_warp", 0.5, uniforms=uniforms)
        self._uniform("radius", 1.0, uniforms=uniforms)
        self._uniform("twist", 0.5, uniforms=uniforms)
        self._uniform("camera_z", 3.0, uniforms=uniforms)
        smooth_k = self._uniform("sdf_smooth_k", 0.25, uniforms=uniforms)

        comb = self._uniform("sdf_combination", 0, kind="int", uniforms=uniforms)

        march = int(self.config.params.get("march_steps", 48))
        march = max(8, min(128, march))
        if self.config.raymarch_steps is not None:
            march = min(march, int(self.config.raymarch_steps))

        sdf_glsl_path = (
            Path(__file__).resolve().parent.parent.parent / "evo" / "fields" / "sdf.glsl"
        )
        sdf_glsl = sdf_glsl_path.read_text() if sdf_glsl_path.exists() else ""

        return ModuleSections(
            header=(
                sdf_glsl + "\n"
                f"#define K_SDF_PRIMITIVE {prim}\n"
                f"#define K_SDF_SHAPE {shape}\n"
                f"#define K_SDF_MARCH {march}\n"
                f"#define K_SDF_COMBINATION ({comb.name})\n"
                f"#define K_SDF_ROUNDNESS ({roundness.name})\n"
                f"#define K_SDF_SURF_NOISE ({surf_noise.name})\n"
                f"#define K_SDF_CAM_DIST ({cam_dist.name})\n"
                f"#define K_SDF_CAM_ORBIT ({cam_orbit.name})\n"
                f"#define K_SDF_LIGHT_ROT ({light_rot.name})\n"
                f"#define K_SDF_CURL ({curl.name})\n"
                f"#define K_SDF_TURB ({turb.name})\n"
                f"#define K_SDF_DOM_WARP ({dom_warp.name})\n"
                f"#define K_SDF_SMOOTH_K ({smooth_k.name})\n"
            ),
            call=(
                "vec3 sdfResult = field_sdf3D(p, t, K_SDF_SHAPE, K_SDF_ROUNDNESS, "
                "K_SDF_SURF_NOISE, K_SDF_CAM_DIST, K_SDF_CAM_ORBIT, K_SDF_LIGHT_ROT, "
                "K_SDF_CURL, K_SDF_TURB, K_SDF_DOM_WARP, K_SDF_MARCH, 0.125, "
                "texture(previous_frame, sample_uv).rgb, K_SDF_COMBINATION, K_SDF_SMOOTH_K);\n"
                "float n;\n"
                "if (sdfResult.x < 0.0) {\n"
                "    n = sdfResult.y;\n"
                "    sdfSurface = vec3(sdfResult.y, sdfResult.z, sdfResult.y * 0.8);\n"
                "} else {\n"
                "    n = sdfResult.y + sdfResult.z;\n"
                "    sdfSurface = vec3(sdfResult.y, sdfResult.z, sdfResult.z * 0.7 + 0.15);\n"
                "}"
            ),
            uniforms=uniforms,
        )
