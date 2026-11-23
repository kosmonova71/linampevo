from __future__ import annotations

from .base import ModuleSections, ModuleUniform, ShaderModule


class BloomPostModule(ShaderModule):
    """HDR bloom effect."""

    def sections(self) -> ModuleSections:
        uniforms: list[ModuleUniform] = []
        threshold = self._uniform("bloom_threshold", 0.6, uniforms=uniforms)
        intensity = self._uniform("bloom_intensity", 0.5, uniforms=uniforms)
        radius = self._uniform("bloom_radius", 1.0, uniforms=uniforms)

        header = """
vec3 bloom(vec3 col, vec2 uv, sampler2D tex, float threshold, float intensity, float radius) {
    vec3 bloom_col = vec3(0.0);
    float count = 0.0;
    float tw = 1.0 / 512.0;

    for (int x = -3; x <= 3; x++)
    for (int y = -3; y <= 3; y++) {
        vec2 offset = vec2(float(x), float(y)) * tw * radius;
        float weight = 1.0 / (1.0 + float(x * x + y * y));
        vec3 s = texture(tex, uv + offset).rgb;
        float br = dot(s, vec3(0.299, 0.587, 0.114));
        if (br > threshold) {
            bloom_col += s * weight;
            count += weight;
        }
    }

    if (count > 0.0) bloom_col /= count;
    return col + bloom_col * intensity;
}
"""

        call = f"col = bloom(col, sample_uv, previous_frame, {threshold.name}, {intensity.name}, {radius.name});"

        return ModuleSections(header=header, call=call, uniforms=uniforms)


class GlowPostModule(ShaderModule):
    """Simple glow / bloom-like effect."""

    def sections(self) -> ModuleSections:
        uniforms: list[ModuleUniform] = []

        header = """
vec3 glow(vec3 col, vec3 glow_col, float intensity) {
    return col + glow_col * intensity * 0.15;
}
"""

        # Glow is handled specially: the call uses the palette output with
        # hue shifted by 0.5. We provide a conditional snippet.
        call = "vec3 glow_col = _pal_col;\ncol = glow(col, glow_col, 1.0 + bassT);"

        return ModuleSections(header=header, call=call, uniforms=uniforms)


class ChromaticAberrationPostModule(ShaderModule):
    """Chromatic aberration / colour fringe."""

    def sections(self) -> ModuleSections:
        uniforms: list[ModuleUniform] = []
        amount = self._uniform("chromatic_aberration_amount", 0.003, uniforms=uniforms)

        call = f"""
{{
    float ca = {amount.name} * (1.0 + bassT * 2.0);
    col.r = texture(previous_frame, sample_uv + vec2(ca, 0.0)).r;
    col.b = texture(previous_frame, sample_uv - vec2(ca, 0.0)).b;
}}
"""

        return ModuleSections(call=call, uniforms=uniforms)


class DistortionPostModule(ShaderModule):
    """UV distortion effect."""

    def sections(self) -> ModuleSections:
        uniforms: list[ModuleUniform] = []
        amount = self._uniform("distortion_amount", 0.5, uniforms=uniforms)
        mode = self._uniform("distortion_mode", 1, kind="int", uniforms=uniforms)

        header = """
vec2 distortion(vec2 uv, float t, float amount, int mode) {
    if (mode == 1) {
        uv += vec2(sin(uv.y * 10.0 + t * 2.0), cos(uv.x * 10.0 + t * 2.0)) * amount * 0.02;
    } else if (mode == 2) {
        float r = length(uv - 0.5);
        uv += (uv - 0.5) * sin(r * 20.0 - t * 3.0) * amount * 0.04;
    }
    return uv;
}
"""

        call = f"sample_uv = distortion(sample_uv, t, {amount.name}, {mode.name});"

        return ModuleSections(header=header, call=call, uniforms=uniforms)


class FeedbackPostModule(ShaderModule):
    """Feedback mixing in post (separate from the main feedback loop)."""

    def sections(self) -> ModuleSections:
        uniforms: list[ModuleUniform] = []
        speed = self._uniform("feedback_speed", 1.0, uniforms=uniforms)
        amount = self._uniform("feedback_amount", 0.5, uniforms=uniforms)
        mode = self._uniform("feedback_post_mode", 0, kind="int", uniforms=uniforms)

        header = """
vec2 feedback_transform(vec2 uv, float t, float speed, int mode, float amount) {
    vec2 c = uv - 0.5;
    float a = 0.03 * sin(t * speed);
    if (mode == 0) {
        c = rot2(a) * c;
    } else if (mode == 1) {
        c = abs(c);
        c = rot2(a) * c;
    }
    c *= 0.97;
    return c + 0.5;
}
"""

        call = (
            f"col = mix(col, texture(previous_frame, "
            f"feedback_transform(sample_uv, t, {speed.name}, {mode.name}, {amount.name})).rgb, 0.3);"
        )

        return ModuleSections(header=header, call=call, uniforms=uniforms)


class PolarGalaxyPostModule(ShaderModule):
    """Polar galaxy operator effect."""

    def sections(self) -> ModuleSections:
        header = """
vec3 postPolarGalaxy(vec3 col, vec2 uv, float t) {
    vec2 c = uv - 0.5;
    float r = length(c);
    float a = atan(c.y, c.x);
    float spiral = sin(a * 5.0 + r * 10.0 - t * 2.0) * 0.5 + 0.5;
    col += vec3(spiral * 0.1, spiral * 0.05, spiral * 0.15);
    return col;
}
"""
        call = "col = postPolarGalaxy(col, sample_uv, t);"
        return ModuleSections(header=header, call=call)


class CellsPostModule(ShaderModule):
    """Cellular / voronoi-style operator effect."""

    def sections(self) -> ModuleSections:
        header = """
vec3 postCells(vec3 col, vec2 uv, float t) {
    vec2 c = uv * 10.0;
    vec2 i = floor(c);
    vec2 f = fract(c);
    vec2 cell = vec2(
        fract(sin(dot(i, vec2(12.9898, 78.233))) * 43758.5453),
        fract(sin(dot(i + 100.0, vec2(12.9898, 78.233))) * 43758.5453)
    );
    float border = 1.0 - smoothstep(0.0, 0.05, min(f.x, 1.0 - f.x) * min(f.y, 1.0 - f.y));
    col += vec3(cell * border * 0.2, border * 0.1);
    return col;
}
"""
        call = "col = postCells(col, sample_uv, t);"
        return ModuleSections(header=header, call=call)


class NonlinearTrailsPostModule(ShaderModule):
    """Nonlinear trails / afterimage operator."""

    def sections(self) -> ModuleSections:
        header = """
vec3 postNonlinearTrails(vec3 col, vec2 uv, float t) {
    vec2 c = uv - 0.5;
    float r = length(c);
    vec3 trail = texture(previous_frame, uv + vec2(sin(r * 5.0 - t), cos(r * 5.0 - t * 0.7)) * 0.02).rgb;
    col = mix(col, trail, 0.15);
    return col;
}
"""
        call = "col = postNonlinearTrails(col, sample_uv, t);"
        return ModuleSections(header=header, call=call)


class AudioTopologyPostModule(ShaderModule):
    """Audio-reactive topology operator."""

    def sections(self) -> ModuleSections:
        header = """
vec3 postAudioTopology(vec3 col, vec2 uv, float t, float bassT, float midT) {
    float warp = bassT * 0.1 + midT * 0.05;
    vec2 duv = uv + vec2(sin(uv.y * 10.0 + t * 2.0), cos(uv.x * 10.0 + t * 2.0)) * warp;
    col = mix(col, texture(previous_frame, duv).rgb, 0.2);
    return col;
}
"""
        call = "col = postAudioTopology(col, sample_uv, t, bassT, midT);"
        return ModuleSections(header=header, call=call)


class DepthCloudPostModule(ShaderModule):
    """Depth cloud / nebula operator."""

    def sections(self) -> ModuleSections:
        header = """
vec3 postDepthCloud(vec3 col, vec2 uv, float t) {
    vec2 c = uv * 3.0;
    float cloud = 0.0;
    float a = 0.5;
    for (int i = 0; i < 4; i++) {
        cloud += a * sin(c.x * 2.0 + c.y * 1.5 + t * 0.3 * a);
        c *= 1.5;
        a *= 0.5;
    }
    col += vec3(cloud * 0.05, cloud * 0.03, cloud * 0.08);
    return col;
}
"""
        call = "col = postDepthCloud(col, sample_uv, t);"
        return ModuleSections(header=header, call=call)


class NeonEdgePostModule(ShaderModule):
    """Neon edge detection operator."""

    def sections(self) -> ModuleSections:
        header = """
vec3 postNeonEdge(vec3 col, vec2 uv, float t) {
    vec2 e = vec2(0.005, 0.0);
    vec3 c0 = texture(previous_frame, uv).rgb;
    vec3 cx = texture(previous_frame, uv + e.xy).rgb;
    vec3 cy = texture(previous_frame, uv + e.yx).rgb;
    float edge = length(cx - c0) + length(cy - c0);
    col += edge * vec3(0.3, 0.1, 0.5) * (0.5 + 0.5 * sin(t));
    return col;
}
"""
        call = "col = postNeonEdge(col, sample_uv, t);"
        return ModuleSections(header=header, call=call)
