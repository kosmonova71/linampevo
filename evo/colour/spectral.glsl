// Spectral Colour Module
// Physics-based spectral rendering
// Genes: wavelength_shift, dispersion

#ifndef SPECTRAL_COLOUR_GLSL
#define SPECTRAL_COLOUR_GLSL

vec3 wavelength_to_rgb(float w, float wavelength_shift) {
    float wl = w + wavelength_shift;
    vec3 col = vec3(0.0);

    if(wl >= 380.0 && wl < 440.0) {
        col.r = -(wl - 440.0) / (440.0 - 380.0);
        col.b = 1.0;
    } else if(wl >= 440.0 && wl < 490.0) {
        col.g = (wl - 440.0) / (490.0 - 440.0);
        col.b = 1.0;
    } else if(wl >= 490.0 && wl < 510.0) {
        col.g = 1.0;
        col.b = -(wl - 510.0) / (510.0 - 490.0);
    } else if(wl >= 510.0 && wl < 580.0) {
        col.r = (wl - 510.0) / (580.0 - 510.0);
        col.g = 1.0;
    } else if(wl >= 580.0 && wl < 645.0) {
        col.r = 1.0;
        col.g = -(wl - 645.0) / (645.0 - 580.0);
    } else if(wl >= 645.0 && wl <= 780.0) {
        col.r = 1.0;
    }

    float factor;
    if(wl >= 380.0 && wl < 420.0) factor = 0.3 + 0.7 * (wl - 380.0) / (420.0 - 380.0);
    else if(wl >= 420.0 && wl <= 700.0) factor = 1.0;
    else if(wl > 700.0 && wl <= 780.0) factor = 0.3 + 0.7 * (780.0 - wl) / (780.0 - 700.0);
    else factor = 0.0;

    return col * factor;
}

vec3 palette_spectral(float t, float wavelength_shift, float dispersion) {
    float base_wl = mix(380.0, 780.0, fract(t * 0.5 + 0.1));
    vec3 col = wavelength_to_rgb(base_wl, wavelength_shift);
    col.r += 0.1 * sin(t * 3.0 + dispersion);
    col.g += 0.1 * sin(t * 3.5 + dispersion + 1.0);
    col.b += 0.1 * sin(t * 4.0 + dispersion + 2.0);
    return clamp(col, 0.0, 1.0);
}

#endif
