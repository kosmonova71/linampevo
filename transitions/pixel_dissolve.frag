#version 130
// Pixel Dissolve Transition
// Uses noise pattern to dissolve between presets

in vec2 v_uv;
varying vec4 gl_FragColor;

uniform sampler2D u_texture_from;
uniform sampler2D u_texture_to;

// Transition uniforms
uniform float iProgressLinear;
uniform float iProgressCosine;
uniform float iProgressBicubic;
uniform float iTransitionDuration;
uniform float iTime;
uniform float iTimeDelta;
uniform float iFrameRate;
uniform int iFrame;
uniform ivec4 iRandStatic;
uniform ivec4 iRandFrame;
uniform float iBass;
uniform float iMid;
uniform float iTreb;
uniform float iBassAtt;
uniform float iMidAtt;
uniform float iTrebAtt;

uniform vec2 fragCoord;
uniform vec3 iResolution;

// Simplex 2D noise
vec3 permute(vec3 x) { return mod(((x*34.0)+1.0)*x, 289.0); }

float snoise(vec2 v){
    const vec4 C = vec4(0.211324865405187, 0.366025403784439,
            -0.577350269189626, 0.024390243902439);
    vec2 i  = floor(v + dot(v, C.yy) );
    vec2 x0 = v -   i + dot(i, C.xx);
    vec2 i1;
    i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    vec4 x12 = x0.xyxy + C.xxzz;
    x12.xy -= i1;
    i = mod(i, 289.0);
    vec3 p = permute( permute( i.y + vec3(0.0, i1.y, 1.0 ))
    + i.x + vec3(0.0, i1.x, 1.0 ));
    vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy),
        dot(x12.zw,x12.zw)), 0.0);
    m = m*m ;
    m = m*m ;
    vec3 x = 2.0 * fract(p * C.www) - 1.0;
    vec3 h = abs(x) - 0.5;
    vec3 ox = floor(x + 0.5);
    vec3 a0 = x - ox;
    m *= 1.79284291400159 - 0.85373472095314 * ( a0*a0 + h*h );
    vec3 g;
    g.x  = a0.x  * x0.x  + h.x  * x0.y;
    g.yz = a0.yz * x12.xz + h.yz * x12.yw;
    return 130.0 * dot(m, g);
}

// Hash function for pseudo-random values
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

void main() {
    vec2 uv = v_uv;
    
    float t = iProgressBicubic;
    
    // Create noise pattern
    // Use static random for consistent pattern during transition
    float seed = float(iRandStatic.x) * 0.01;
    
    // Combine multiple noise octaves
    float noise = 0.0;
    noise += snoise(uv * 8.0 + seed) * 0.5;
    noise += snoise(uv * 16.0 + seed * 1.5) * 0.25;
    noise += snoise(uv * 32.0 + seed * 2.0) * 0.125;
    noise = noise * 0.5 + 0.5;  // Normalize to 0-1
    
    // Add audio reactivity to noise
    noise += iBass * 0.1 * sin(iTime * 4.0 + uv.x * 10.0);
    noise += iMid * 0.05 * cos(iTime * 3.0 + uv.y * 15.0);
    noise = clamp(noise, 0.0, 1.0);
    
    // Sample both textures
    vec4 from_color = texture(u_texture_from, uv);
    vec4 to_color = texture(u_texture_to, uv);
    
    // Pixel dissolve threshold
    // As t increases, more pixels switch from to to from
    float threshold = t;
    
    // Add edge softness
    float edge = 0.1 + iTreb * 0.05;
    float mix_factor = smoothstep(threshold - edge, threshold + edge, noise);
    
    // Audio can create "bursts" of transition
    float burst = step(0.8, iBass) * 0.2 * sin(iTime * 10.0);
    mix_factor = clamp(mix_factor + burst, 0.0, 1.0);
    
    gl_FragColor = mix(from_color, to_color, mix_factor);
}
