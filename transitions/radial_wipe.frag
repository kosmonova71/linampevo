#version 130
// Radial Wipe Transition
// New preset expands from center as a growing circle

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

void main() {
    vec2 uv = v_uv;
    
    // Center coordinates
    vec2 center = vec2(0.5);
    vec2 centered_uv = uv - center;
    
    // Calculate distance from center
    float dist = length(centered_uv);
    
    // Maximum distance (corner to center)
    float max_dist = length(vec2(0.5));
    
    // Use bicubic progress for smoother animation
    float t = iProgressBicubic;
    
    // Add subtle audio reactivity - bass makes the circle pulse slightly
    float pulse = iBass * 0.02 * sin(iTime * 8.0);
    float threshold = t * max_dist + pulse;
    
    // Sample both textures
    vec4 from_color = texture(u_texture_from, uv);
    vec4 to_color = texture(u_texture_to, uv);
    
    // Create smooth edge with cosine falloff
    float edge_width = 0.02 + iMid * 0.01;  // Width affected by mid frequencies
    float mix_factor = smoothstep(threshold - edge_width, threshold + edge_width, dist);
    
    // Note: We reverse the mix because we want the circle to grow from center
    // When t=0, we want full from_color (dist > threshold -> mix_factor = 1 -> from)
    // When t=1, we want full to_color (dist < threshold -> mix_factor = 0 -> to)
    gl_FragColor = mix(to_color, from_color, mix_factor);
}
