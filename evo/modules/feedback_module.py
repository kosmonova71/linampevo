from __future__ import annotations

from .base import ModuleSections, ShaderModule


class DriftFeedbackModule(ShaderModule):
    """Slow drift/noise displacement feedback."""

    def sections(self) -> ModuleSections:
        return ModuleSections(
            header="""
vec2 feedbackFn(vec2 uv, float t) {
    vec2 c = uv - 0.5;
    c += vec2(sin(t * 0.1 + c.y * 2.0) * 0.02, cos(t * 0.15 + c.x * 2.0) * 0.02);
    return c + 0.5;
}
"""
        )


class MirrorFeedbackModule(ShaderModule):
    """Mirror reflection feedback."""

    def sections(self) -> ModuleSections:
        return ModuleSections(
            header="""
vec2 feedbackFn(vec2 uv, float t) {
    vec2 c = uv - 0.5;
    c = abs(c);
    c *= 0.98 + 0.01 * sin(t * 0.3);
    return c + 0.5;
}
"""
        )


class SpiralFeedbackModule(ShaderModule):
    """Spiral rotate + breathe feedback."""

    def sections(self) -> ModuleSections:
        return ModuleSections(
            header="""
vec2 feedbackFn(vec2 uv, float t) {
    vec2 c = uv - 0.5;
    float a = 0.05 * sin(t * 0.3) + 0.02;
    c = rot2(a) * c;
    c *= 1.0 + 0.01 * sin(t * 0.5);
    return c + 0.5;
}
"""
        )


class ZoomFeedbackModule(ShaderModule):
    """Zoom-in feedback (pull toward centre)."""

    def sections(self) -> ModuleSections:
        return ModuleSections(
            header="""
vec2 feedbackFn(vec2 uv, float t) {
    vec2 c = uv - 0.5;
    float zoom = 0.97 + 0.02 * sin(t * 0.2);
    c *= zoom;
    return c + 0.5;
}
"""
        )


class KaleidoFeedbackModule(ShaderModule):
    """Kaleidoscope symmetry feedback."""

    def sections(self) -> ModuleSections:
        sym = int(self.config.params.get("kaleido_sym", 6))
        sym = max(2, min(16, sym))

        return ModuleSections(
            header=f"""
#define K_KALEIDO_SYM {sym}

vec2 feedbackFn(vec2 uv, float t) {{
    vec2 c = uv - 0.5;
    float r = length(c);
    float a = atan(c.y, c.x);
    a = mod(a, EVO_TAU / K_KALEIDO_SYM);
    a = abs(a - EVO_TAU / (2.0 * K_KALEIDO_SYM));
    c = vec2(cos(a), sin(a)) * r;
    c *= 1.0 + 0.01 * sin(t * 0.4);
    return c + 0.5;
}}
"""
        )
