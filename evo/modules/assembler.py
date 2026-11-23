#!/usr/bin/env python3

"""ShaderAssembler — builds a complete GLSL fragment shader from a genome.

The assembler:
  1. Consults the :class:`ModuleRegistry` to instantiate the correct modules.
  2. Collects :class:`ModuleSections` from each module.
  3. Wraps them in the GLSL shader template (version, uniforms, varying, main).

This replaces the monolithic ``generate()`` method in
:class:`ModularShaderFactory` with a clean, composable architecture.
"""

from __future__ import annotations

import logging
import math
from typing import Any, ClassVar

from .base import (
    ModuleConfig,
    ModuleSections,
)
from .common_module import CommonModule
from .module_source_cache import ModuleFingerprint, ModuleSourceCache, _section_hash
from .registry import ModuleRegistry, get_default_registry

logger = logging.getLogger(__name__)

_TWO_PI = 6.28318


class ShaderAssembler:
    """Assembles a complete GLSL fragment shader from genome selections.

    Usage::

        assembler = ShaderAssembler()
        source, uniforms = assembler.build(genome, ...)
    """

    def __init__(
        self, registry: ModuleRegistry | None = None, source_cache: ModuleSourceCache | None = None
    ):
        self.registry = registry or get_default_registry()
        self.source_cache = source_cache
        self._last_fingerprint: ModuleFingerprint | None = None

    @property
    def last_fingerprint(self) -> ModuleFingerprint | None:
        return self._last_fingerprint

    def _obtain_sections(
        self,
        category: str,
        name: str,
        params: dict[str, Any] | None = None,
        fingerprint: ModuleFingerprint | None = None,
        **kwargs,
    ) -> ModuleSections | None:
        """Resolve module sections from cache or generate fresh."""
        cache = self.source_cache
        mod = self.registry.create(category, name, params, **kwargs)
        if mod is None:
            return None
        key = cache.key_for(category, name, params) if cache else None
        sections = cache.get(key) if (cache and key) else None
        if sections is None:
            sections = mod.sections()
            if cache and key:
                cache.put(key, sections)
        if fingerprint is not None:
            fingerprint.update(category, _section_hash(sections))
        return sections

    # ---- Public API ----------------------------------------------------

    def build(
        self,
        genome,
        brightness: float = 1.0,
        feedback_alpha: float = 0.85,
        morphology: dict | None = None,
        raymarch_steps: int | None = None,
        fractal_layers: int | None = None,
        history_depth: int = 0,
    ) -> tuple[str, dict[str, dict[str, Any]]]:
        """Build a complete GLSL fragment shader from a genome.

        Args:
            genome: ModularGenome instance with field_type, warp, palette, etc.
            brightness: Global brightness multiplier.
            feedback_alpha: Feedback blending factor.
            morphology: Optional dict of procedural biology metrics.
            raymarch_steps: Cap on raymarch iterations (quality tier).
            fractal_layers: Cap on fractal layers (quality tier).
            history_depth: Number of temporal history frames.

        Returns:
            ``(source, uniforms)`` where ``source`` is the complete GLSL
            fragment shader source and ``uniforms`` maps uniform name ->
            ``{"value", "kind"}``.
        """
        # Collect sections from all modules -------------------------------------------------
        fingerprint = ModuleFingerprint()

        # 1. Common module (always included)
        common = CommonModule(ModuleConfig("common", "common", {}))
        all_sections = common.sections()
        fingerprint.update("common", _section_hash(all_sections))

        # 2. Field module
        field_call = ""
        field_sections = self._obtain_sections(
            "field",
            genome.field_type,
            genome.field_params,
            fingerprint=fingerprint,
            raymarch_steps=raymarch_steps,
            fractal_layers=fractal_layers,
        )
        if field_sections is not None:
            all_sections.merge(field_sections)
            field_call = field_sections.call
            if field_call:
                all_sections.call = all_sections.call.replace(field_call, "").strip()
        else:
            logger.warning("Field module '%s' not found, using fallback", genome.field_type)
            field_call = "float n = 0.5 + 0.5 * sin(p.x * 3.0 + p.y * 5.0 + t);\n"
            all_sections.call = ""

        # 3. Warp module
        warp_sections = self._obtain_sections(
            "warp",
            genome.warp,
            genome.warp_params,
            fingerprint=fingerprint,
        )
        if warp_sections is not None:
            all_sections.merge(warp_sections)

        # 4. Palette module
        pal_sections = self._obtain_sections(
            "palette",
            genome.palette,
            genome.palette_params,
            fingerprint=fingerprint,
        )
        if pal_sections is not None:
            all_sections.merge(pal_sections)
            pal_call = pal_sections.call
            if pal_call:
                all_sections.call = all_sections.call.replace(pal_call, "").strip()
        else:
            # Default cosine palette
            pal_sections = self._obtain_sections(
                "palette",
                "cosine",
                {},
                fingerprint=fingerprint,
            )
            if pal_sections is not None:
                all_sections.merge(pal_sections)
                pal_call = pal_sections.call
                if pal_call:
                    all_sections.call = all_sections.call.replace(pal_call, "").strip()

        for post_name in genome.post:
            post_sections = self._obtain_sections(
                "post",
                post_name,
                genome.post_params,
                fingerprint=fingerprint,
            )
            if post_sections is not None:
                # Glow is handled specially via {glow_code} in the template
                if post_name == "glow":
                    all_sections.merge(
                        ModuleSections(
                            header=post_sections.header,
                            footer=post_sections.footer,
                            uniforms=post_sections.uniforms,
                        )
                    )
                else:
                    all_sections.merge(post_sections)
            else:
                logger.warning("Post module '%s' not found", post_name)

        # 6. Feedback module
        fb_sections = self._obtain_sections(
            "feedback",
            genome.feedback_mode,
            genome.field_params,
            fingerprint=fingerprint,
        )
        if fb_sections is not None:
            all_sections.merge(fb_sections)
        else:
            # Default spiral feedback
            fb_sections = self._obtain_sections(
                "feedback",
                "spiral",
                {},
                fingerprint=fingerprint,
            )
            if fb_sections is not None:
                all_sections.merge(fb_sections)
            else:
                all_sections.header += "vec2 feedbackFn(vec2 uv, float t) { return uv; }\n"

        # 7. Animation module (Milkdrop-style per-frame animation)
        anim_params = getattr(genome, "animation", {})
        if anim_params:
            anim_sections = self._obtain_sections(
                "animation",
                "default",
                anim_params,
                fingerprint=fingerprint,
            )
            if anim_sections is not None:
                all_sections.merge(anim_sections)

        # 8. Morphology module (optional)
        morph_sections = ModuleSections()
        if morphology:
            morph_sections = (
                self._obtain_sections(
                    "morph",
                    "morphology",
                    morphology,
                    fingerprint=fingerprint,
                )
                or morph_sections
            )
            if morph_sections:
                all_sections.merge(morph_sections)

        # 8. Lighting module (optional)
        if getattr(genome, "lighting", "none") != "none":
            light_sections = self._obtain_sections(
                "pass",
                "lighting",
                getattr(genome, "lighting_params", {}),
                fingerprint=fingerprint,
            )
            if light_sections is not None:
                all_sections.merge(light_sections)

        # 9. Camera module (optional)
        if getattr(genome, "camera", "none") != "none":
            cam_sections = self._obtain_sections(
                "pass",
                "camera",
                getattr(genome, "camera_params", {}),
                fingerprint=fingerprint,
            )
            if cam_sections is not None:
                all_sections.merge(cam_sections)

        # 10. Particle module (optional)
        if getattr(genome, "particles", "none") != "none":
            part_sections = self._obtain_sections(
                "pass",
                "particle",
                getattr(genome, "particle_params", {}),
                fingerprint=fingerprint,
            )
            if part_sections is not None:
                all_sections.merge(part_sections)

        # ---- Build the template -------------------------------------------------

        is_sdf = genome.field_type == "sdf"

        # Feedback alpha and brightness as literals
        feedback_lit = self._glsl_float(feedback_alpha, lo=0.0, hi=0.999)
        brightness_lit = self._glsl_float(brightness, lo=0.0, hi=10.0)

        # Uniform declarations
        uniform_decls = "\n".join(u.glsl_decl for u in all_sections.uniforms if u.name)

        # History uniforms
        history_uniforms = ""
        if history_depth > 0:
            history_uniforms = self._build_history_uniforms(history_depth)

        # Morphology feedback injection value
        morph_feedback = "0.0"
        if morphology:
            regen_val = morphology.get("regeneration", 0.5)
            morph_feedback = f"({regen_val} * 0.08)"

        # Build hue and colour assignment
        pal_call_var = "_pal_col"
        palette_call_expr = f"vec3 {pal_call_var} = palette_cosine(hueT, 0, 0.0, t);"
        if genome.palette == "cosine":
            palette_call_expr = f"vec3 {pal_call_var} = palette_cosine(hueT, K_PAL_ID, K_PAL_SHIFT, t * K_PAL_SPEED);"
        elif genome.palette == "spectral":
            palette_call_expr = f"vec3 {pal_call_var} = palette_spectral(hueT, K_SPEC_WL_SHIFT, K_SPEC_DISPERSION);"
        elif genome.palette == "plasma":
            palette_call_expr = f"vec3 {pal_call_var} = palette_plasma(hueT, K_PLASMA_SPEED, K_PLASMA_COMPLEXITY, K_PLASMA_SAT);"

        # SDF vs regular colour
        sdf_col_assign = ""
        regular_col_assign = f"col = {pal_call_var};"
        if is_sdf:
            sdf_col_assign = "col = sdfSurface;"
            regular_col_assign = ""

        # Build field evaluation line
        if is_sdf:
            field_eval_section = field_call
        else:
            # The field call snippet from the module is in field_call
            field_eval_section = field_call

        # Glow post injection (special handling: uses palette output)
        has_glow = "glow" in genome.post
        glow_code = ""
        if has_glow:
            glow_code = f"""
    // Glow
    {{
        vec3 glow_col = {pal_call_var};
        col = glow(col, glow_col, 1.0 + bassT);
    }}
"""

        # Assemble the GLSL source
        source = f"""#version 330 core
in vec2 v_uv;
out vec4 fragColor;

uniform float u_time;
uniform vec2 u_resolution;

uniform float bass;
uniform float mid;
uniform float treble;

uniform sampler2D previous_frame;
{history_uniforms}

{uniform_decls}

{all_sections.header}

bool bad(vec3 c)
{{
    return any(isnan(c)) || any(isinf(c));
}}

void main() {{
    vec2 uv = v_uv;
    vec2 sample_uv = v_uv;
    
    // Apply Milkdrop-style animation transform
    uv = apply_anim_transform(uv);
    sample_uv = apply_anim_transform(sample_uv);
    
    uv = (uv - 0.5) * vec2(u_resolution.x / u_resolution.y, 1.0);

    float t = u_time;

    // Symmetry
    float r = length(uv);
    float th = atan(uv.y, uv.x);
    th = abs(mod(th, {_TWO_PI} / 4.0) - {_TWO_PI} / 8.0);

    vec2 p = vec2(cos(th), sin(th)) * r;

    // Warp
    p = apply_warp(p, t);

    // SDF surface output
    vec3 sdfSurface = vec3(0.0);

    // Field evaluation
    {field_eval_section}

    // Audio reactivity
    float bassT = clamp(bass, 0.0, 1.0);
    float midT = clamp(mid, 0.0, 1.0);
    float trebT = clamp(treble, 0.0, 1.0);

    n += 0.10 * sin(p.x * 3.0 + t * 0.9 + trebT * 2.0);
    n += 0.05 * sin(p.y * 2.0 + t * 1.1 + bassT * 1.5);
    n = clamp(n, 0.0, 1.0);

    // Morphology field injection
    {morph_sections.call}

    // Colour
    float hueT = n
        + 0.10 * t
        + 0.25 * bassT
        + 0.15 * midT
        + 0.20 * trebT
        + 0.05 * (bassT + midT + trebT) * 0.5;

    {palette_call_expr}

    // Colour buffer
    vec3 col = vec3(0.0);

    {regular_col_assign}
    {sdf_col_assign}

    // Morphology colour injection
    {morph_sections.footer}

    // Post-processing calls (from post module sections)
    {all_sections.call}

    // Glow (special)
    {glow_code}

    // Feedback (transform selected by genome.feedback_mode)
    vec2 fb_uv = feedbackFn(sample_uv, t);
    vec3 prev = texture(previous_frame, fb_uv).rgb;
    float a = clamp({feedback_lit} - {morph_feedback}, 0.0, 0.999);
    col = mix(col, prev, a);
    {self._build_temporal_history(history_depth) if history_depth > 0 else ""}

    // Footer sections (post-module final adjustments)
    {all_sections.footer}

    // Brightness
    col *= {brightness_lit};

    // NaN / Inf guard
    if(bad(col)) {{
        col = vec3(0.0);
    }}

    col = clamp(col, 0.0, 1.0);
    fragColor = vec4(col, 1.0);
}}
"""

        # Build uniform metadata dict for the renderer
        uniforms_dict: dict[str, dict[str, Any]] = {}
        for u in all_sections.uniforms:
            uniforms_dict[u.name] = {"value": u.value, "kind": u.kind}

        self._last_fingerprint = fingerprint
        return source, uniforms_dict

    # ---- Per-pass shader generation -----------------------------------------

    _BLEND_ID: ClassVar[dict[str, int]] = {
        "replace": 0,
        "alpha": 1,
        "add": 2,
        "multiply": 3,
        "screen": 4,
    }
    _FB_ID: ClassVar[dict[str, int]] = {"none": 0, "self": 1, "previous": 2, "first": 3}

    def build_for_pass(
        self,
        genome,
        pass_type: str = "field",
        blend_mode: str = "replace",
        feedback_route: str = "none",
        brightness: float = 1.0,
        feedback_alpha: float = 0.85,
        morphology: dict | None = None,
        history_depth: int = 0,
    ) -> tuple[str, dict[str, dict[str, Any]]]:
        """Build a standalone per-pass fragment shader for the multi-pass pipeline.

        Unlike :meth:`build`, which produces a single complete shader that reads
        from ``previous_frame``, this method generates a shader that reads from
        ``u_input_tex`` (the previous pass's output) and optionally blends with
        it.  The renderer compiles one program per pass and chains FBOs.

        Args:
            genome: The per-pass ``ModularGenome``.
            pass_type: One of ``field``, ``post``, ``lighting``, ``particle``,
                ``composite``.  Determines which modules are included.
            blend_mode: How this pass blends with ``u_input_tex``.
            feedback_route: ``none`` / ``self`` / ``previous`` / ``first``.
            brightness: Global brightness multiplier.
            feedback_alpha: Feedback mix strength.
            morphology: Optional biology metrics.
            history_depth: Temporal echo history depth (currently unused in
                per-pass mode to keep shaders compact).

        Returns:
            ``(source, uniforms)`` tuple.
        """
        from evo.pipeline_genome import PassGenome  # noqa: F401  (type helper)

        # ---- Module selection -------------------------------------------------
        all_sections = ModuleSections()
        common = CommonModule(ModuleConfig("common", "common", {}))
        all_sections.merge(common.sections())
        fingerprint = ModuleFingerprint()
        fingerprint.update("common", _section_hash(common.sections()))

        pass_type = (pass_type or "field").lower()
        is_field_pass = pass_type == "field"
        is_sdf_pass = pass_type == "sdf"
        is_rd_pass = pass_type == "reaction_diffusion"
        is_feedback_pass = pass_type == "feedback"
        is_post_pass = pass_type == "post"
        is_lighting_pass = pass_type == "lighting"
        is_particle_pass = pass_type == "particle"
        is_volume_pass = pass_type == "volume"
        is_composite_pass = pass_type == "composite"

        # Field-producing passes: generate a procedural colour field
        if is_field_pass or is_sdf_pass or is_rd_pass:
            # For SDF and reaction_diffusion pass types, the genome's
            # field_type may not match — build a temporary params dict.
            if is_sdf_pass:
                # Force SDF field regardless of genome field_type
                field_type_for_pass = "sdf"
                fp = dict(genome.field_params)
                fp.setdefault("sdf_shape", 0)
                fp.setdefault("sdf_roundness", 0.5)
                fp.setdefault("march_steps", 48)
            elif is_rd_pass:
                field_type_for_pass = "reaction_diffusion"
                fp = dict(genome.field_params)
                fp.setdefault("feed_rate", 0.055)
                fp.setdefault("kill_rate", 0.062)
                fp.setdefault("diffusion_u", 1.0)
                fp.setdefault("diffusion_v", 0.5)
                fp.setdefault("iterations", 32)
            else:
                field_type_for_pass = genome.field_type
                fp = genome.field_params

            field_sections = self._obtain_sections(
                "field",
                field_type_for_pass,
                fp,
                fingerprint=fingerprint,
            )
            field_call = ""
            if field_sections is not None:
                all_sections.merge(field_sections)
                field_call = field_sections.call
                if field_call:
                    all_sections.call = all_sections.call.replace(field_call, "").strip()
            else:
                logger.warning("Field module '%s' not found, using fallback", field_type_for_pass)
                field_call = "float n = 0.5 + 0.5 * sin(p.x * 3.0 + p.y * 5.0 + t);\n"
                all_sections.call = ""

            # Warp (only for field and sdf passes)
            if is_field_pass or is_sdf_pass:
                warp_sections = self._obtain_sections(
                    "warp",
                    genome.warp,
                    genome.warp_params,
                    fingerprint=fingerprint,
                )
                if warp_sections is not None:
                    all_sections.merge(warp_sections)

            # Palette
            pal_name = genome.palette
            pal_sections = self._obtain_sections(
                "palette",
                pal_name,
                genome.palette_params,
                fingerprint=fingerprint,
            )
            if pal_sections is not None:
                all_sections.merge(pal_sections)
            else:
                pal_sections = self._obtain_sections(
                    "palette",
                    "cosine",
                    {},
                    fingerprint=fingerprint,
                )
                if pal_sections is not None:
                    all_sections.merge(pal_sections)

            # Feedback (only for field and RD passes; SDF disables)
            if not is_sdf_pass:
                fb_sections = self._obtain_sections(
                    "feedback",
                    genome.feedback_mode,
                    fp,
                    fingerprint=fingerprint,
                )
                if fb_sections is not None:
                    all_sections.merge(fb_sections)
                else:
                    fb_sections = self._obtain_sections(
                        "feedback",
                        "spiral",
                        {},
                        fingerprint=fingerprint,
                    )
                    if fb_sections is not None:
                        all_sections.merge(fb_sections)
                    else:
                        all_sections.header += "vec2 feedbackFn(vec2 uv, float t) { return uv; }\n"

        # Post modules: present for all pass types
        for post_name in genome.post:
            post_sections = self._obtain_sections(
                "post",
                post_name,
                genome.post_params,
                fingerprint=fingerprint,
            )
            if post_sections is not None:
                # Glow is handled specially via {glow_code} in the template
                if post_name == "glow":
                    all_sections.merge(
                        ModuleSections(
                            header=post_sections.header,
                            footer=post_sections.footer,
                            uniforms=post_sections.uniforms,
                        )
                    )
                else:
                    if not is_post_pass and not is_lighting_pass:
                        post_sections.call = post_sections.call.replace(
                            "previous_frame", "u_input_tex"
                        )
                    all_sections.merge(post_sections)
            else:
                logger.warning("Post module '%s' not found", post_name)

        # Pass-level modules
        if is_lighting_pass:
            pass_sections = self._obtain_sections(
                "pass",
                "lighting",
                genome.field_params,
                fingerprint=fingerprint,
            )
            if pass_sections is not None:
                all_sections.merge(pass_sections)
        elif is_particle_pass:
            pass_sections = self._obtain_sections(
                "pass",
                "particle",
                genome.field_params,
                fingerprint=fingerprint,
            )
            if pass_sections is not None:
                all_sections.merge(pass_sections)
        elif is_volume_pass or is_composite_pass:
            pass_sections = self._obtain_sections(
                "pass",
                "volume",
                {},
                fingerprint=fingerprint,
            )
            if pass_sections is not None:
                all_sections.merge(pass_sections)

        # Morphology
        if morphology:
            morph_sections = self._obtain_sections(
                "morph",
                "morphology",
                morphology,
                fingerprint=fingerprint,
            )
            if morph_sections is not None:
                all_sections.merge(morph_sections)

        # ---- Template constants ----------------------------------------------
        is_field_type = is_field_pass or is_sdf_pass or is_rd_pass

        # Animation module (Milkdrop-style per-frame animation: zoom/rot/warp).
        # Field-producing passes carry the animated view transform so the
        # multi-pass evolution animates like MilkDrop; non-field passes skip it.
        # ``has_anim_module`` gates template emission of apply_anim_transform()
        # so a load failure degrades to static rendering instead of a shader
        # compile error.
        has_anim_module = False
        if is_field_type:
            anim_params = getattr(genome, "animation", {})
            if anim_params:
                anim_sections = self._obtain_sections(
                    "animation",
                    "default",
                    anim_params,
                    fingerprint=fingerprint,
                )
                if anim_sections is not None:
                    all_sections.merge(anim_sections)
                    has_anim_module = True

        brightness_lit = self._glsl_float(brightness, lo=0.0, hi=10.0)
        fb_alpha_lit = self._glsl_float(feedback_alpha, lo=0.0, hi=0.999)

        blend_id = self._BLEND_ID.get(blend_mode, 0)
        fb_route_id = self._FB_ID.get(feedback_route, 0)

        uniform_decls = "\n".join(u.glsl_decl for u in all_sections.uniforms if u.name)

        # History uniforms (per-pass inherits depth from caller)
        history_uniforms = ""
        if history_depth > 0:
            history_uniforms = self._build_history_uniforms(history_depth)

        # Palette variable name (used in glow special handling)
        pal_call_var = "_pal_col"

        # Colour init
        regular_col_assign = "col = texture(u_input_tex, v_uv).rgb;"
        sdf_col_assign = ""
        if is_field_pass:
            regular_col_assign = f"col = {pal_call_var};"
        elif is_sdf_pass:
            sdf_col_assign = "col = sdfSurface;"
            regular_col_assign = ""
        elif is_rd_pass:
            regular_col_assign = f"col = {pal_call_var};"
        elif is_feedback_pass:
            regular_col_assign = "col = texture(u_input_tex, v_uv).rgb;"

        # Field eval section (only for field-producing passes)
        field_eval_section = ""
        if is_field_pass or is_sdf_pass or is_rd_pass:
            field_eval_section = field_call

        # Glow special handling
        has_glow = "glow" in genome.post
        glow_code = ""
        if has_glow and is_field_pass:
            glow_code = f"""
    // Glow
    {{
        vec3 glow_col = {pal_call_var};
        col = glow(col, glow_col, 1.0 + bassT);
    }}
"""

        # Audio reactivity (field-producing passes only)
        audio_reactivity = ""
        if is_field_type:
            audio_reactivity = """
    // Audio reactivity
    float bassT = clamp(bass, 0.0, 1.0);
    float midT = clamp(mid, 0.0, 1.0);
    float trebT = clamp(treble, 0.0, 1.0);

    n += 0.10 * sin(p.x * 3.0 + t * 0.9 + trebT * 2.0);
    n += 0.05 * sin(p.y * 2.0 + t * 1.1 + bassT * 1.5);
    n = clamp(n, 0.0, 1.0);

    // Colour
    float hueT = n
        + 0.10 * t
        + 0.25 * bassT
        + 0.15 * midT
        + 0.20 * trebT
        + 0.05 * (bassT + midT + trebT) * 0.5;
"""

        source = f"""#version 330 core
in vec2 v_uv;
out vec4 fragColor;

uniform float u_time;
uniform vec2 u_resolution;

uniform float bass;
uniform float mid;
uniform float treble;

uniform sampler2D u_input_tex;
uniform sampler2D u_feedback_tex;
uniform int u_input_enabled;
uniform int u_blend_mode;
uniform float u_blend_alpha;
uniform int u_feedback_route;
uniform float u_feedback_alpha;
{history_uniforms if history_depth > 0 else ""}

{uniform_decls}

{all_sections.header}

bool bad(vec3 c)
{{
    return any(isnan(c)) || any(isinf(c));
}}

void main() {{
    vec2 uv = v_uv;
    vec2 sample_uv = v_uv;
    {"    // Apply Milkdrop-style per-frame animation transform" if has_anim_module else ""}
    {"    uv = apply_anim_transform(uv);" if has_anim_module else ""}
    {"    sample_uv = apply_anim_transform(sample_uv);" if has_anim_module else ""}

    float t = u_time;

{"    // ---- Domain warp (field-producing passes only) ----" if is_field_type else ""}
{"    uv = (uv - 0.5) * vec2(u_resolution.x / u_resolution.y, 1.0);" if is_field_type else ""}
{"    float r = length(uv);" if is_field_type else ""}
{"    float th = atan(uv.y, uv.x);" if is_field_type else ""}
{"    th = abs(mod(th, " + repr(_TWO_PI) + " / 4.0) - " + repr(_TWO_PI) + " / 8.0);" if is_field_type else ""}
{"    vec2 p = vec2(cos(th), sin(th)) * r;" if is_field_type else ""}
{"    p = apply_warp(p, t);" if is_field_type else ""}
{"    vec3 sdfSurface = vec3(0.0);" if is_field_type else ""}

{"    // Field evaluation" if is_field_type else ""}
    {field_eval_section if is_field_type else ""}

{audio_reactivity}

    {regular_col_assign}

    {sdf_col_assign}

    // Post-processing calls (from post module sections)
    {all_sections.call}

    // Glow (special, field-pass only)
{"    " + glow_code.strip() if is_field_pass else ""}

    // Input blending (multi-pass)
    if (u_input_enabled == 1) {{
        vec3 input_col = texture(u_input_tex, sample_uv).rgb;
        if (u_blend_mode == 0) {{ /* replace — keep col */ }}
        else if (u_blend_mode == 1) {{ col = mix(input_col, col, u_blend_alpha); }}
        else if (u_blend_mode == 2) {{ col = input_col + col; }}
        else if (u_blend_mode == 3) {{ col = input_col * col; }}
        else if (u_blend_mode == 4) {{ col = 1.0 - (1.0 - input_col) * (1.0 - col); }}
    }}

    // Feedback mixing (non-self routes use u_feedback_tex)
    if (u_feedback_route == 1) {{
        vec2 fb_uv = feedbackFn(sample_uv, t);
        vec3 prev = texture(u_input_tex, fb_uv).rgb;
        col = mix(col, prev, u_feedback_alpha);
    }} else if (u_feedback_route == 2 || u_feedback_route == 3) {{
        vec2 fb_uv = feedbackFn(sample_uv, t);
        vec3 prev = texture(u_feedback_tex, fb_uv).rgb;
        col = mix(col, prev, u_feedback_alpha);
    }}

    // Brightness
    col *= {brightness_lit};

    // NaN / Inf guard
    if(bad(col)) {{
        col = vec3(0.0);
    }}

    col = clamp(col, 0.0, 1.0);
    fragColor = vec4(col, 1.0);
}}
"""

        uniforms_dict: dict[str, dict[str, Any]] = {}
        for u in all_sections.uniforms:
            uniforms_dict[u.name] = {"value": u.value, "kind": u.kind}

        uniforms_dict["u_input_enabled"] = {"value": 1, "kind": "int"}
        uniforms_dict["u_blend_mode"] = {"value": blend_id, "kind": "int"}
        uniforms_dict["u_blend_alpha"] = {
            "value": 0.85 if blend_mode == "alpha" else 0.0,
            "kind": "float",
        }
        uniforms_dict["u_feedback_route"] = {"value": fb_route_id, "kind": "int"}
        uniforms_dict["u_feedback_alpha"] = {"value": fb_alpha_lit, "kind": "float"}

        self._last_fingerprint = fingerprint
        return source, uniforms_dict

    def build_pipeline_per_pass(
        self,
        pipeline,
        brightness: float = 1.0,
        feedback_alpha: float = 0.85,
        morphology: dict | None = None,
        history_depth: int = 0,
    ) -> list[tuple[str, dict[str, dict[str, Any]]]]:
        """Generate one shader per pipeline pass for true multi-FBO rendering.

        Returns:
            A list of ``(source, uniforms)`` tuples, one per ``PassGenome``
            in the pipeline, in render order.
        """
        results: list[tuple[str, dict[str, dict[str, Any]]]] = []
        for p in pipeline.passes:
            try:
                src, unis = self.build_for_pass(
                    genome=p.modules,
                    pass_type=p.pass_type,
                    blend_mode=p.blend_mode,
                    feedback_route=p.feedback_route,
                    brightness=brightness,
                    feedback_alpha=feedback_alpha,
                    morphology=morphology,
                    history_depth=history_depth,
                )
                results.append((src, unis))
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("Per-pass shader generation failed for %s: %s", p.pass_type, exc)
        return results

    # ---- Pipeline build (single-shader concatenation, retained for compat) -----

    def build_pipeline(
        self,
        pipeline,
        brightness: float = 1.0,
        feedback_alpha: float = 0.85,
        morphology: dict | None = None,
        history_depth: int = 0,
    ) -> tuple[str, dict[str, dict[str, Any]]]:
        """Build a multi-pass pipeline shader from a PipelineGenome.

        This is a simplified version that concatenates pass sections into
        a single shader.  For true multi-FBO support, the renderer should
        split into separate programs per pass.
        """
        # Use the primary field pass for the field/warp/palette/feedback core,
        # then add post/lighting/particle passes as additional sections.
        from evo.pipeline_genome import PassGenome

        all_sections = ModuleSections()
        common = CommonModule(ModuleConfig("common", "common", {}))
        all_sections.merge(common.sections())
        fingerprint = ModuleFingerprint()
        fingerprint.update("common", _section_hash(common.sections()))

        # Find the field pass
        field_pass = None
        for p in pipeline.passes:
            if p.pass_type == "field":
                field_pass = p
                break
        if field_pass is None:
            field_pass = pipeline.passes[0] if pipeline.passes else PassGenome()

        # Build field pass sections
        fg = field_pass.modules
        field_sections = self._obtain_sections(
            "field",
            fg.field_type,
            fg.field_params,
            fingerprint=fingerprint,
        )
        field_call = ""
        if field_sections is not None:
            all_sections.merge(field_sections)
            field_call = field_sections.call
            if field_call:
                all_sections.call = all_sections.call.replace(field_call, "").strip()

        warp_sections = self._obtain_sections(
            "warp",
            fg.warp,
            fg.warp_params,
            fingerprint=fingerprint,
        )
        if warp_sections is not None:
            all_sections.merge(warp_sections)

        pal_sections = self._obtain_sections(
            "palette",
            fg.palette,
            fg.palette_params,
            fingerprint=fingerprint,
        )
        if pal_sections is not None:
            all_sections.merge(pal_sections)
            pal_call = pal_sections.call
            if pal_call:
                all_sections.call = all_sections.call.replace(pal_call, "").strip()

        fb_sections = self._obtain_sections(
            "feedback",
            fg.feedback_mode,
            fg.field_params,
            fingerprint=fingerprint,
        )
        if fb_sections is not None:
            all_sections.merge(fb_sections)

        for post_name in fg.post:
            post_sections = self._obtain_sections(
                "post",
                post_name,
                fg.post_params,
                fingerprint=fingerprint,
            )
            if post_sections is not None:
                if post_name == "glow":
                    all_sections.merge(
                        ModuleSections(
                            header=post_sections.header,
                            footer=post_sections.footer,
                            uniforms=post_sections.uniforms,
                        )
                    )
                else:
                    all_sections.merge(post_sections)

        # Morphology
        if morphology:
            morph_sections = self._obtain_sections(
                "morph",
                "morphology",
                morphology,
                fingerprint=fingerprint,
            )
            if morph_sections is not None:
                all_sections.merge(morph_sections)

        # Build pass-level sections
        for p in pipeline.passes:
            if p.pass_type == "lighting":
                pass_sections = self._obtain_sections(
                    "pass",
                    "lighting",
                    p.modules.field_params,
                    fingerprint=fingerprint,
                )
                if pass_sections is not None:
                    all_sections.merge(pass_sections)
            elif p.pass_type == "particle":
                pass_sections = self._obtain_sections(
                    "pass",
                    "particle",
                    p.modules.field_params,
                    fingerprint=fingerprint,
                )
                if pass_sections is not None:
                    all_sections.merge(pass_sections)
            elif p.pass_type == "composite":
                pass_sections = self._obtain_sections(
                    "pass",
                    "volume",
                    pipeline.global_params,
                    fingerprint=fingerprint,
                )
                if pass_sections is not None:
                    all_sections.merge(pass_sections)

        # Build the template similar to build()
        feedback_lit = self._glsl_float(feedback_alpha, lo=0.0, hi=0.999)
        brightness_lit = self._glsl_float(brightness, lo=0.0, hi=10.0)

        uniform_decls = "\n".join(u.glsl_decl for u in all_sections.uniforms if u.name)

        history_uniforms = ""
        if history_depth > 0:
            history_uniforms = self._build_history_uniforms(history_depth)

        pal_call_var = "_pal_col"
        if fg.palette == "cosine":
            palette_call = f"vec3 {pal_call_var} = palette_cosine(hueT, K_PAL_ID, K_PAL_SHIFT, t * K_PAL_SPEED);"
        elif fg.palette == "spectral":
            palette_call = f"vec3 {pal_call_var} = palette_spectral(hueT, K_SPEC_WL_SHIFT, K_SPEC_DISPERSION);"
        elif fg.palette == "plasma":
            palette_call = f"vec3 {pal_call_var} = palette_plasma(hueT, K_PLASMA_SPEED, K_PLASMA_COMPLEXITY, K_PLASMA_SAT);"
        else:
            palette_call = f"vec3 {pal_call_var} = palette_cosine(hueT, 0, 0.0, t);"

        source = f"""#version 330 core
in vec2 v_uv;
out vec4 fragColor;

uniform float u_time;
uniform vec2 u_resolution;

uniform float bass;
uniform float mid;
uniform float treble;

uniform sampler2D previous_frame;
uniform sampler2D feedback_buffer;
{history_uniforms}

{uniform_decls}

{all_sections.header}

bool bad(vec3 c)
{{
    return any(isnan(c)) || any(isinf(c));
}}

void main() {{
    vec2 uv = v_uv;
    vec2 sample_uv = v_uv;
    uv = (uv - 0.5) * vec2(u_resolution.x / u_resolution.y, 1.0);

    float t = u_time;

    // Symmetry
    float r = length(uv);
    float th = atan(uv.y, uv.x);
    th = abs(mod(th, {_TWO_PI} / 4.0) - {_TWO_PI} / 8.0);

    vec2 p = vec2(cos(th), sin(th)) * r;

    // Warp
    p = apply_warp(p, t);

    vec3 sdfSurface = vec3(0.0);

    // Field
    {field_call}

    float bassT = clamp(bass, 0.0, 1.0);
    float midT = clamp(mid, 0.0, 1.0);
    float trebT = clamp(treble, 0.0, 1.0);

    n += 0.10 * sin(p.x * 3.0 + t * 0.9 + trebT * 2.0);
    n += 0.05 * sin(p.y * 2.0 + t * 1.1 + bassT * 1.5);
    n = clamp(n, 0.0, 1.0);

    float hueT = n
        + 0.10 * t
        + 0.25 * bassT
        + 0.15 * midT
        + 0.20 * trebT
        + 0.05 * (bassT + midT + trebT) * 0.5;

    {palette_call}

    vec3 col = vec3(0.0);
    col = {pal_call_var};

    {all_sections.footer}

    // Feedback
    vec2 fb_uv = feedbackFn(sample_uv, t);
    vec3 prev = texture(previous_frame, fb_uv).rgb;
    float a = clamp({feedback_lit}, 0.0, 0.999);
    col = mix(col, prev, a);
    {self._build_temporal_history(history_depth) if history_depth > 0 else ""}

    col *= {brightness_lit};

    if(bad(col)) {{
        col = vec3(0.0);
    }}

    col = clamp(col, 0.0, 1.0);
    fragColor = vec4(col, 1.0);
}}
"""

        uniforms_dict: dict[str, dict[str, Any]] = {}
        for u in all_sections.uniforms:
            uniforms_dict[u.name] = {"value": u.value, "kind": u.kind}

        self._last_fingerprint = fingerprint
        return source, uniforms_dict

    # ---- Helpers -------------------------------------------------------

    @staticmethod
    def _glsl_float(value: float, lo: float | None = None, hi: float | None = None) -> str:
        value = float(value)
        if math.isnan(value):  # NaN check
            value = 0.0
        if lo is not None:
            value = max(lo, value)
        if hi is not None:
            value = min(hi, value)
        s = repr(value)
        if "." not in s and "e" not in s and "E" not in s:
            s += ".0"
        return s

    @staticmethod
    def _build_history_uniforms(depth: int) -> str:
        depth = max(0, min(5, int(depth)))
        lines = []
        for i in range(depth):
            lines.append(f"uniform sampler2D previous_frame_{i + 1};")
        return "\n".join(lines)

    @staticmethod
    def _build_temporal_history(depth: int) -> str:
        depth = max(0, min(5, int(depth)))
        if depth == 0:
            return ""
        weights = [0.20, 0.15, 0.10, 0.07, 0.04]
        terms = []
        for i in range(depth):
            w = weights[i] if i < len(weights) else 0.02
            terms.append(f"texture(previous_frame_{i + 1}, v_uv).rgb * {w}")
        expr = " + ".join(terms)
        return f""" {{
    vec3 temporal_echo = {expr};
    col += temporal_echo;
}}"""
