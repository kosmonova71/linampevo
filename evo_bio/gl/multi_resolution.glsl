#version 330 core
// evo_bio/gl/multi_resolution.glsl
//
// Multi-resolution rendering pipeline (point 9). Each stage runs at its own
// scale; the final post-processing runs at full resolution. This file is the
// shared header / sampler declarations used by per-stage programs. The Python
// side (evo_bio.gpu.MultiResolutionPipeline) allocates one ping-pong FBO per
// stage at the sizes below and chains them:
//
//   1/8 res  -> reaction diffusion (expensive, benefits most from low res)
//   1/4 res  -> feedback / trails
//   1/2 res  -> main field generation
//   full res -> post processing (bloom, chromatic, glow)
//
// Painter's order is defined by MULTI_RES_PIPELINE in evo_bio/gpu.py.

uniform sampler2D u_rd;        // reaction-diffusion result (1/8)
uniform sampler2D u_feedback;  // feedback/trails result (1/4)
uniform sampler2D u_field;     // main field result (1/2)
uniform sampler2D u_prev;      // previous full-res frame
uniform float u_time;
uniform vec2 u_resolution;
uniform vec4 u_audio;

// Upsample a low-res stage to the current resolution with bilinear sampling.
vec3 sample_stage(sampler2D stage, vec2 uv){
    return texture(stage, uv).rgb;
}
