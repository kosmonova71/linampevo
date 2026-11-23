#version 330 core
// evo_bio/gl/compute_fitness.glsl
// GPU reduction pass: folds a small rendered frame into a single fitness float.
// Mirrors evo_bio.fitness (entropy / edges / motion / colour variance) entirely
// on the GPU so only one float crosses the bus (point 8).
in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_frame;     // current frame (small res)
uniform sampler2D u_prev;      // previous frame for motion
uniform vec2 u_resolution;

float luma(vec3 c){ return dot(c, vec3(0.299,0.587,0.114)); }

void main(){
    vec3 c = texture(u_frame, v_uv).rgb;
    float l = luma(c);

    // Local entropy proxy: variance within a 3x3 neighbourhood.
    float v = 0.0, n = 0.0;
    for(int y=-1;y<=1;y++){
        for(int x=-1;x<=1;x++){
            vec2 off = vec2(float(x),float(y))/u_resolution;
            float ll = luma(texture(u_frame, v_uv+off).rgb);
            v += (ll-l)*(ll-l);
            n += 1.0;
        }
    }
    float edge = sqrt(v/max(n,1.0));

    // Motion vs previous frame.
    float m = abs(l - luma(texture(u_prev, v_uv).rgb));

    // Colour variance proxy.
    float cv = length(c - vec3(l));

    // Composite (weights match evo_bio.fitness.DEFAULT_WEIGHTS reasonably).
    float fitness = 0.12*clamp(l,0.0,1.0)
                  + 0.10*clamp(edge*10.0,0.0,1.0)
                  + 0.07*clamp(m*15.0,0.0,1.0)
                  + 0.08*clamp(cv*20.0,0.0,1.0);

    // Penalise blown-out brightness.
    fitness -= max(l-0.8,0.0)*0.3;

    fragColor = vec4(clamp(fitness,0.0,1.0), 0.0, 0.0, 1.0);
}
