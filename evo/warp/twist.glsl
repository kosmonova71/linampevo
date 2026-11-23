// Twist Warp Module
// Barrel/pincushion distortion
// Genes: amount, iterations

#ifndef TWIST_WARP_GLSL
#define TWIST_WARP_GLSL

vec2 warp_twist(vec2 p, float t, float amount, int iterations) {
    float total = 0.0;
    for(int i = 0; i < 4; i++) {
        if(i >= iterations) break;
        float r = length(p);
        float th = atan(p.y, p.x);
        float twist = amount * (1.0 - smoothstep(0.0, 1.5, r)) * sin(t * 0.2);
        float cs = cos(twist);
        float sn = sin(twist);
        p = vec2(p.x * cs - p.y * sn, p.x * sn + p.y * cs);
        total += twist;
    }
    return p;
}

#endif
