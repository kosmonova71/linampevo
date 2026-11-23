#version 130
// Transition Vertex Shader
// Standard vertex shader for all transition effects

attribute vec2 position;
#define 0;
attribute vec2 texcoord;
#define 1;

varying vec2 v_uv;

void main() {
    v_uv = texcoord;
    gl_Position = vec4(position, 0.0, 1.0);
}
