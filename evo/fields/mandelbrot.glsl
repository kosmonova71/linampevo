// Mandelbrot Field Module
// Fractal distance estimation
// Genes: max_iter, escape_radius, zoom, julia_mode, julia_c

#ifndef MANDELBROT_FIELD_GLSL
#define MANDELBROT_FIELD_GLSL

float field_mandelbrot(vec2 p, float t, int max_iter, float escape_radius,
                       float zoom, int julia_mode, float julia_cx, float julia_cy) {
    float angle = t * 0.05;
    float ca = cos(angle);
    float sa = sin(angle);
    mat2 rot = mat2(ca, -sa, sa, ca);
    vec2 z = rot * p / zoom;
    vec2 c;

    if(julia_mode == 1) {
        z = p / zoom;
        c = vec2(julia_cx, julia_cy);
    } else {
        c = p / zoom;
    }

    float iter = 0.0;
    for(int i = 0; i < 128; i++) {
        if(i >= max_iter) break;
        float x = z.x * z.x - z.y * z.y + c.x;
        float y = 2.0 * z.x * z.y + c.y;
        z = vec2(x, y);
        if(dot(z, z) > escape_radius * escape_radius) break;
        iter += 1.0;
    }

    if(iter >= float(max_iter) - 0.5) return 0.0;

    // Smooth iteration count
    float log_zn = log(dot(z, z)) * 0.5;
    float nu = log(log_zn / log(escape_radius)) / log(2.0);
    iter = iter + 1.0 - nu;

    return iter / float(max_iter);
}

#endif
