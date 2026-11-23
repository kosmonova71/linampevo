// evo/post/cells.glsl
// Reaction-diffusion illusion (trick 3): no simulation needed, just two
// decorrelated fbm layers differenced into a smoothstep cell mask. Reads as
// bacteria, coral or alien skin laid over the organism.
vec3 postCells(vec3 col, vec2 uv, float t) {
    vec2 q = uv * 6.0;
    float a = fbm(q + t * 0.05, 4, 2.0, 0.5);
    float b = fbm(q * 2.0 - t * 0.03, 4, 2.0, 0.5);
    float mask = smoothstep(0.45, 0.55, a - b);
    return mix(col, vec3(mask), 0.35);
}
