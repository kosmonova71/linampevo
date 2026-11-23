// evo/post/depth_cloud.glsl
// Fake raymarch depth (trick 8): sum several offset resamples of the
// previous frame to build volume - clouds and nebula depth from 2D fields.
vec3 postDepthCloud(vec3 col, vec2 uv, float t) {
    vec3 depth = vec3(0.0);
    for(int i = 0; i < 6; i++) {
        float f = float(i) * 0.01;
        depth += texture(previous_frame, uv + vec2(f, f * 0.5)).rgb;
    }
    depth /= 6.0;
    return col + (depth - col) * 0.4;
}
