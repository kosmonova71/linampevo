// Bloom Post-Processing Module
// HDR bloom effect
// Genes: threshold, intensity, radius

#ifndef BLOOM_POST_GLSL
#define BLOOM_POST_GLSL

vec3 bloom(vec3 col, vec2 uv, sampler2D tex, float threshold, float intensity, float radius) {
    vec3 bloom_col = vec3(0.0);
    float count = 0.0;
    float tw = 1.0 / 512.0;

    for(int x = -3; x <= 3; x++)
    for(int y = -3; y <= 3; y++) {
        vec2 offset = vec2(float(x), float(y)) * tw * radius;
        float weight = 1.0 / (1.0 + float(x * x + y * y));
        vec3 s = texture(tex, uv + offset).rgb;
        float br = dot(s, vec3(0.299, 0.587, 0.114));
        if(br > threshold) {
            bloom_col += s * weight;
            count += weight;
        }
    }

    if(count > 0.0) bloom_col /= count;
    return col + bloom_col * intensity;
}

#endif
