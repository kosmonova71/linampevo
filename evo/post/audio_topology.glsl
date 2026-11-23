// evo/post/audio_topology.glsl
// Audio-driven topology mutation (trick 7): instead of only changing colour,
// let the music warp the feedback sample so the organism's "body" reshapes to
// the beat. Bass scales, treble shears, mid mixes.
vec3 postAudioTopology(vec3 col, vec2 uv, float t) {
    vec2 q = uv;
    q.x += sin(uv.y * 20.0 + t) * treble * 0.05;
    q.y += cos(uv.x * 16.0 - t) * bass * 0.05;
    q *= 1.0 + bass * 0.15;
    vec3 warped = texture(previous_frame, q).rgb;
    return mix(col, warped, 0.3 + mid * 0.2);
}
