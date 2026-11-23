// Cosine Colour Module
// Classic cosine-based palette (Inigo Quilez technique)
// Genes: palette_id, shift_speed, colour_shift

#ifndef COSINE_COLOUR_GLSL
#define COSINE_COLOUR_GLSL

vec3 palette_cosine(float t, int palette_id, float colour_shift, float t_mod) {
    float shift = t_mod * colour_shift;
    vec3 a, b, c, d;

    if(palette_id == 0) {
        a = vec3(0.15, 0.20, 0.35);
        b = vec3(0.65, 0.25, 0.75);
        c = vec3(1.00, 0.95, 0.85);
        d = vec3(0.20, 0.40, 0.60);
    } else if(palette_id == 1) {
        a = vec3(0.05, 0.10, 0.20);
        b = vec3(0.90, 0.40, 0.20);
        c = vec3(2.00, 1.80, 1.60);
        d = vec3(0.00, 0.33, 0.67);
    } else if(palette_id == 2) {
        a = vec3(0.50, 0.50, 0.50);
        b = vec3(0.50, 0.50, 0.50);
        c = vec3(1.00, 1.00, 1.00);
        d = vec3(0.00, 0.33, 0.67);
    } else if(palette_id == 3) {
        a = vec3(0.80, 0.20, 0.20);
        b = vec3(0.20, 0.60, 0.80);
        c = vec3(2.00, 1.00, 1.50);
        d = vec3(0.00, 0.10, 0.20);
    } else if(palette_id == 4) {
        a = vec3(0.10, 0.30, 0.10);
        b = vec3(0.70, 0.90, 0.30);
        c = vec3(1.50, 2.00, 1.00);
        d = vec3(0.00, 0.50, 0.00);
    } else {
        a = vec3(0.20, 0.00, 0.30);
        b = vec3(0.80, 0.60, 0.90);
        c = vec3(1.00, 1.50, 2.00);
        d = vec3(0.30, 0.00, 0.50);
    }

    return a + b * cos(6.28318 * (c * t + d) + vec3(shift, shift + 2.094, shift + 4.189));
}

#endif
