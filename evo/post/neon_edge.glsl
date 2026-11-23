// evo/post/neon_edge.glsl
// Neon edge extraction (trick 9): local gradient of the previous frame
// becomes glowing veins, electric networks and biological structures.
vec3 postNeonEdge(vec3 col, vec2 uv, float t) {
    float e = 0.0035;
    vec3 a = texture(previous_frame, uv + vec2(e, 0.0)).rgb;
    vec3 b = texture(previous_frame, uv - vec2(e, 0.0)).rgb;
    vec3 c = texture(previous_frame, uv + vec2(0.0, e)).rgb;
    vec3 d = texture(previous_frame, uv - vec2(0.0, e)).rgb;
    float edge = length(a - b) + length(c - d);
    return col + vec3(edge) * 2.0;
}
