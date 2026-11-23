// evo/post/glow.glsl
// Analytic glow: brighten the brightest channels to fake a halo.
vec3 postGlow(vec3 col, vec2 uv, float t) {
    float l = max(max(col.r, col.g), col.b);
    vec3 halo = col * pow(l, 2.0);
    return col + halo * 0.6;
}
