// Plasma Colour Module
// Classic plasma effect colours
// Genes: speed, complexity, saturation

#ifndef PLASMA_COLOUR_GLSL
#define PLASMA_COLOUR_GLSL

vec3 palette_plasma(float t, float speed, float complexity, float saturation) {
    float r = sin(t * complexity * 3.14159 + speed) * 0.5 + 0.5;
    float g = sin(t * complexity * 3.14159 + speed + 2.094) * 0.5 + 0.5;
    float b = sin(t * complexity * 3.14159 + speed + 4.189) * 0.5 + 0.5;

    vec3 col = vec3(r, g, b);
    float gray = dot(col, vec3(0.299, 0.587, 0.114));
    col = mix(vec3(gray), col, saturation);
    return col;
}

#endif
