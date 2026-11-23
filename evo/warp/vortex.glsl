// Vortex Warp Module
// Spiral/vortex coordinate transformation
// Genes: strength, arms, twist_rate

#ifndef VORTEX_WARP_GLSL
#define VORTEX_WARP_GLSL

vec2 warp_vortex(vec2 p, float t, float strength, float arms, float twist_rate) {
    float r = length(p);
    float th = atan(p.y, p.x);
    float vortex_angle = strength * r * (1.0 + twist_rate * sin(th * arms + t * 0.3));
    float cs = cos(vortex_angle);
    float sn = sin(vortex_angle);
    mat2 V = mat2(cs, -sn, sn, cs);
    return V * p;
}

#endif
