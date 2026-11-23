// SDF / Distance Field Module — Full 3D Organic Worlds
// Signed distance field geometry rendering with organic deformation,
// multi-shape organisms, orbit camera, and rich lighting.
// Genes: primitive_type, radius, sdf_shape, sdf_roundness,
//        surface_noise, camera_distance, camera_orbit_speed,
//        light_rotation, march_steps, feedback_amount

#ifndef SDF_FIELD_GLSL
#define SDF_FIELD_GLSL

// ---------------------------------------------------------------------------
// 3D hash / noise helpers (self-contained; do not depend on evo/common)
// ---------------------------------------------------------------------------
float hash31(vec3 p) {
    p = fract(p * vec3(443.8975, 397.2973, 491.1871));
    p += dot(p, p.yzx + 19.19);
    return fract((p.x + p.y) * p.z);
}

float noise3D(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);

    return mix(
        mix(
            mix(hash31(i), hash31(i + vec3(1,0,0)), f.x),
            mix(hash31(i + vec3(0,1,0)), hash31(i + vec3(1,1,0)), f.x),
            f.y
        ),
        mix(
            mix(hash31(i + vec3(0,0,1)), hash31(i + vec3(1,0,1)), f.x),
            mix(hash31(i + vec3(0,1,1)), hash31(i + vec3(1,1,1)), f.x),
            f.y
        ),
        f.z
    );
}

float fbm3D(vec3 p) {
    float v = 0.0;
    float a = 0.5;
    vec3 shift = vec3(100.0);
    for (int i = 0; i < 4; i++) {
        v += a * noise3D(p);
        p = p * 2.0 + shift;
        a *= 0.5;
    }
    return v;
}

// ---------------------------------------------------------------------------
// Primitive SDFs
// ---------------------------------------------------------------------------
float sdSphere(vec3 p, float r) {
    return length(p) - r;
}

float sdRoundBox(vec3 p, vec3 b, float r) {
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
}

float sdTorus(vec3 p, vec2 t) {
    vec2 q = vec2(length(p.xz) - t.x, p.y);
    return length(q) - t.y;
}

float sdOctahedron(vec3 p, float s) {
    p = abs(p);
    return (p.x + p.y + p.z - s) * 0.57735027;
}

float sdCapsule(vec3 p, float h, float r) {
    p.y -= clamp(p.y, 0.0, h);
    return length(p) - r;
}

float sdCylinder(vec3 p, float h, float r) {
    vec2 d = abs(vec2(length(p.xz), p.y)) - vec2(r, h);
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}

float sdCross(vec3 p) {
    float da = max(abs(p.x), abs(p.y));
    float db = max(abs(p.y), abs(p.z));
    float dc = max(abs(p.z), abs(p.x));
    return min(da, min(db, dc));
}

float sdBox(vec3 p, vec3 b) {
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

float sdMengerSponge(vec3 p, float s) {
    float d = sdBox(p, vec3(s));
    float s1 = s;
    for (int i = 0; i < 3; i++) {
        vec3 a = mod(p * 2.0, 2.0) - 1.0;
        s1 *= 3.0;
        vec3 r = 1.0 - 3.0 * abs(a);
        float c = sdCross(r) / s1;
        d = max(d, c);
    }
    return d;
}

float sdKleinBottle(vec3 p, float s) {
    float x = p.x;
    float y = p.y;
    float z = p.z;
    float d = length(vec2(
        (x*x + y*y + z*z + 2.0*y*z - s*s) * (x*x + y*y + z*z - 2.0*y*z - s*s) - 4.0*s*s*(x*x + y*y),
        (x*x + y*y + z*z - 2.0*y*z - s*s) * 2.0*x*z - 4.0*s*s*x*z
    )) - 0.0;
    return d * 0.1;
}

float sdGyroid(vec3 p, float scale, float thickness, float bias) {
    p *= scale;
    return abs(dot(sin(p), cos(p.zxy)) - bias) / scale - thickness;
}

float mandelbulb(vec3 p, float power, int iterations) {
    vec3 z = p;
    float dr = 1.0;
    float r = 0.0;
    for (int i = 0; i < 8; i++) {
        if (i >= iterations) break;
        r = length(z);
        if (r > 2.0) break;
        float theta = acos(z.z / r);
        float phi = atan(z.y, z.x);
        dr = pow(r, power - 1.0) * power * dr + 1.0;
        float zr = pow(r, power);
        theta = theta * power;
        phi = phi * power;
        z = zr * vec3(sin(theta) * cos(phi), sin(phi) * sin(theta), cos(theta));
        z += p;
    }
    return 0.5 * log(r) * r / dr;
}

float hypertexture(vec3 p, float t, float density, float scale) {
    float n = 0.0;
    float amp = 1.0;
    vec3 pp = p * scale;
    for (int i = 0; i < 4; i++) {
        n += amp * (sin(pp.x + t) * sin(pp.y + t * 0.7) * sin(pp.z + t * 1.3));
        pp *= 2.0;
        amp *= 0.5;
    }
    return n * density;
}

float reactionDiffusion3D(vec3 p, float t, float feed, float kill, float diffU, float diffV) {
    float u = 0.5 + 0.1 * sin(p.x * 3.0 + t);
    float v = 0.25 + 0.05 * cos(p.y * 4.0 + t * 0.8);
    float dt = 0.1;
    for (int i = 0; i < 6; i++) {
        float lapU = 0.0;
        float lapV = 0.0;
        float h = 0.1;
        lapU += 0.5 * sin(p.x * 3.0 + t + h) + 0.5 * sin(p.x * 3.0 + t - h) - u;
        lapV += 0.5 * cos(p.y * 4.0 + t * 0.8 + h) + 0.5 * cos(p.y * 4.0 + t * 0.8 - h) - v;
        float uvv = u * v * v;
        u += (diffU * lapU - uvv + feed * (1.0 - u)) * dt;
        v += (diffV * lapV + uvv - (kill + feed) * v) * dt;
        u = clamp(u, 0.0, 1.0);
        v = clamp(v, 0.0, 1.0);
    }
    return v;
}

// ---------------------------------------------------------------------------
// Boolean / smooth operators
// ---------------------------------------------------------------------------
float opUnion(float a, float b) {
    return min(a, b);
}

float opSubtract(float a, float b) {
    return max(a, -b);
}

float opIntersect(float a, float b) {
    return max(a, b);
}

float opSmoothUnion(float d1, float d2, float k) {
    float h = clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0);
    return mix(d2, d1, h) - k * h * (1.0 - h);
}

float opSmoothSubtract(float d1, float d2, float k) {
    float h = clamp(0.5 - 0.5 * (d2 + d1) / k, 0.0, 1.0);
    return mix(d2, -d1, h) + k * h * (1.0 - h);
}

float opSmoothIntersect(float d1, float d2, float k) {
    float h = clamp(0.5 - 0.5 * (d2 - d1) / k, 0.0, 1.0);
    return mix(d2, d1, h) + k * h * (1.0 - h);
}

float opRound(float d, float r) {
    return d - r;
}

// ---------------------------------------------------------------------------
// Organic deformation
// ---------------------------------------------------------------------------
vec3 deform(vec3 p, float t, float curl_strength, float turbulence, float domain_warp) {
    vec3 q = p;

    // Domain warping — pushes space in organic swirls
    q += sin(q.yzx * 1.8 + t * 0.7) * domain_warp * 0.25;
    q += cos(q.zxy * 2.3 + t * 0.5) * domain_warp * 0.15;

    // Turbulence — finer detail noise displacement
    float n = fbm3D(q * 1.5 + t * 0.3);
    q += (n - 0.5) * turbulence * 0.4;

    // Curl / vortex twist
    float angle = curl_strength * 0.6 * sin(q.y * 2.5 + t * 0.8);
    float c = cos(angle);
    float s = sin(angle);
    q.xz = mat2(c, -s, s, c) * q.xz;

    return q;
}

// ---------------------------------------------------------------------------
// Surface noise (bumps / skin)
// ---------------------------------------------------------------------------
float surfaceNoise(vec3 p, float amount) {
    return (fbm3D(p * 6.0) - 0.5) * amount;
}

// ---------------------------------------------------------------------------
// Single primitive map (selected by gene)
// ---------------------------------------------------------------------------
float primitive(vec3 p, int prim, float radius, float roundness) {
    float r = radius;
    float rn = roundness;
    if (prim == 0) {
        return sdSphere(p, r);
    } else if (prim == 1) {
        return sdRoundBox(p, vec3(r), rn);
    } else if (prim == 2) {
        return sdTorus(p, vec2(r, r * 0.35));
    } else if (prim == 3) {
        return sdOctahedron(p, r);
    } else if (prim == 4) {
        return sdCapsule(p, r * 1.6, r * 0.45);
    } else if (prim == 5) {
        return sdCylinder(p, r * 1.2, r * 0.6);
    } else if (prim == 6) {
        return sdMengerSponge(p, r);
    } else if (prim == 7) {
        return sdKleinBottle(p, r);
    } else if (prim == 8) {
        return sdGyroid(p, 2.0 / max(r, 0.01), 0.02, 0.0);
    } else if (prim == 9) {
        return mandelbulb(p, 8.0, 6);
    }
    return sdSphere(p, r);
}

// ---------------------------------------------------------------------------
// Organism map — multi-shape combination
// Combines a central body with orbiting satellites and deformations.
// ---------------------------------------------------------------------------
float organism(vec3 p, float t,
               int shape, float roundness, float surf_noise,
               float curl, float turb, float dom_warp,
               int combination, float smooth_k) {

    vec3 dp = deform(p, t, curl, turb, dom_warp);

    float d = primitive(dp, shape, 1.0, roundness);

    // Central body surface noise
    d += surfaceNoise(dp, surf_noise * 0.08);

    // Satellite 1 — orbiting cell / moon
    vec3 s1 = dp - vec3(sin(t * 1.1) * 1.35, cos(t * 0.9) * 0.6, sin(t * 0.7) * 1.1);
    float d1 = primitive(s1, (shape + 1) % 10, 0.38, roundness * 0.8);
    d1 += surfaceNoise(s1, surf_noise * 0.05);

    // Satellite 2 — second moon
    vec3 s2 = dp - vec3(cos(t * 0.8) * 1.5, sin(t * 1.3) * 0.5, cos(t * 1.0) * 0.9);
    float d2 = primitive(s2, (shape + 2) % 10, 0.28, roundness * 0.6);
    d2 += surfaceNoise(s2, surf_noise * 0.04);

    // Core hole / eye — subtracts a small sphere
    float hole = sdSphere(dp - vec3(sin(t * 1.4) * 0.3, cos(t * 1.2) * 0.4, 0.0), 0.22);

    // Evolved combination
    if (combination == 0) {
        // Union default
        d = opUnion(opUnion(d, d1), d2);
        d = opSmoothSubtract(hole, d, 0.15);
    } else if (combination == 1) {
        // Smooth blend all
        d = opSmoothUnion(d, d1, smooth_k);
        d = opSmoothUnion(d, d2, smooth_k);
        d = opSmoothSubtract(hole, d, smooth_k * 0.6);
    } else if (combination == 2) {
        // Intersect / carve
        d = opIntersect(d, d1);
        d = opSmoothUnion(d, d2, smooth_k);
        d = opSmoothSubtract(hole, d, smooth_k * 0.6);
    } else if (combination == 3) {
        // Subtract satellites
        d = opSmoothUnion(d, d1, smooth_k);
        d = opSmoothSubtract(d2, d, smooth_k);
        d = opSmoothSubtract(hole, d, smooth_k * 0.6);
    } else {
        // Default
        d = opUnion(opUnion(d, d1), d2);
        d = opSmoothSubtract(hole, d, 0.15);
    }

    return d;
}

// ---------------------------------------------------------------------------
// Normal estimation
// ---------------------------------------------------------------------------
vec3 calcNormal(vec3 p, float t,
                int shape, float roundness, float surf_noise,
                float curl, float turb, float dom_warp,
                int combination, float smooth_k) {
    float e = 0.002;
    return normalize(vec3(
        organism(p + vec3(e,0,0), t, shape, roundness, surf_noise, curl, turb, dom_warp, combination, smooth_k)
      - organism(p - vec3(e,0,0), t, shape, roundness, surf_noise, curl, turb, dom_warp, combination, smooth_k),
        organism(p + vec3(0,e,0), t, shape, roundness, surf_noise, curl, turb, dom_warp, combination, smooth_k)
      - organism(p - vec3(0,e,0), t, shape, roundness, surf_noise, curl, turb, dom_warp, combination, smooth_k),
        organism(p + vec3(0,0,e), t, shape, roundness, surf_noise, curl, turb, dom_warp, combination, smooth_k)
      - organism(p - vec3(0,0,e), t, shape, roundness, surf_noise, curl, turb, dom_warp, combination, smooth_k)
    ));
}

// ---------------------------------------------------------------------------
// Soft shadow
// ---------------------------------------------------------------------------
float softShadow(vec3 ro, vec3 rd, float t_max, float k,
                 int shape, float roundness, float surf_noise,
                 float curl, float turb, float dom_warp,
                 int combination, float smooth_k) {
    float res = 1.0;
    float d = 0.02;
    for (int i = 0; i < 24; i++) {
        float h = organism(ro + rd * d, t_max, shape, roundness, surf_noise, curl, turb, dom_warp, combination, smooth_k);
        res = min(res, k * h / d);
        d += clamp(h, 0.02, 0.2);
        if (h < 0.001 || d > 8.0) break;
    }
    return clamp(res, 0.0, 1.0);
}

// ---------------------------------------------------------------------------
// Ambient occlusion
// ---------------------------------------------------------------------------
float ambientOcclusion(vec3 p, vec3 n, float t,
                       int shape, float roundness, float surf_noise,
                       float curl, float turb, float dom_warp,
                       int combination, float smooth_k) {
    float ao = 0.0;
    float w = 1.0;
    for (int i = 0; i < 5; i++) {
        float d = 0.05 + 0.15 * float(i);
        float h = organism(p + n * d, t, shape, roundness, surf_noise, curl, turb, dom_warp, combination, smooth_k);
        ao += (d - h) * w;
        w *= 0.55;
    }
    return clamp(1.0 - 2.5 * ao, 0.0, 1.0);
}

// ---------------------------------------------------------------------------
// Raymarch — finds 3D surface
// ---------------------------------------------------------------------------
float raymarch(vec3 ro, vec3 rd, float t,
               int shape, float roundness, float surf_noise,
               float curl, float turb, float dom_warp,
               int steps, int combination, float smooth_k) {
    float depth = 0.0;
    for (int i = 0; i < 128; i++) {
        if (i >= steps) break;
        vec3 p = ro + rd * depth;
        float dist = organism(p, t, shape, roundness, surf_noise, curl, turb, dom_warp, combination, smooth_k);
        if (dist < 0.001) return depth;
        depth += dist;
        if (depth > 30.0) break;
    }
    return -1.0;
}

// ---------------------------------------------------------------------------
// Main field function — full 3D render
// Returns vec3(depth, diffuse, specular) for the post pipeline.
// ---------------------------------------------------------------------------
vec3 field_sdf3D(vec2 uv, float t,
               int shape, float roundness, float surf_noise,
               float cam_dist, float cam_orbit, float light_rot,
               float curl, float turb, float dom_warp,
               int march_steps,
               float feedback_amount,
               vec3 prev_col,
               int combination, float smooth_k) {

    // ---- Camera orbit ----
    float ca = cam_orbit * t * 0.4;
    vec3 ro = vec3(
        sin(ca) * cam_dist,
        1.2 + sin(t * 0.35) * 0.6,
        cos(ca) * cam_dist
    );
    vec3 target = vec3(0.0);

    vec3 fwd = normalize(target - ro);
    vec3 right = normalize(cross(fwd, vec3(0.0, 1.0, 0.0)));
    vec3 up = cross(right, fwd);
    vec3 rd = normalize(fwd * 1.8 + right * uv.x + up * uv.y);

    // ---- Light orbit ----
    float la = light_rot * t * 0.25;
    vec3 lightDir = normalize(vec3(sin(la), 0.8, cos(la)));
    vec3 lightDir2 = normalize(vec3(-sin(la * 0.7), 0.5, -cos(la * 0.7)));

    // ---- Raymarch ----
    float d = raymarch(ro, rd, t, shape, roundness, surf_noise,
                       curl, turb, dom_warp, march_steps, combination, smooth_k);

    // ---- Background — nebula / gradient with previous-frame blend ----
    float nebula = 0.5 + 0.5 * sin(uv.x * 4.0 + t * 0.3) * sin(uv.y * 4.0 + t * 0.2);
    vec3 bg = mix(vec3(0.02, 0.01, 0.04), vec3(0.06, 0.02, 0.10), nebula);

    // Feedback trail blend
    bg = mix(bg, prev_col, feedback_amount * 0.35);

    if (d < 0.0) {
        return vec3(-1.0, bg.r, bg.g);
    }

    // ---- Shading ----
    vec3 pos = ro + rd * d;
    vec3 n = calcNormal(pos, t, shape, roundness, surf_noise, curl, turb, dom_warp, combination, smooth_k);

    // Primary light
    float diff = max(dot(n, lightDir), 0.0);
    float spec = pow(max(dot(reflect(-lightDir, n), -rd), 0.0), 48.0);

    // Fill light
    float diff2 = max(dot(n, lightDir2), 0.0) * 0.25;

    // Ambient occlusion
    float ao = ambientOcclusion(pos, n, t, shape, roundness, surf_noise, curl, turb, dom_warp, combination, smooth_k);

    // Soft shadow
    float shadow = softShadow(pos + n * 0.01, lightDir, t, 8.0,
                              shape, roundness, surf_noise, curl, turb, dom_warp, combination, smooth_k);

    // Combine
    float lighting = (diff * shadow + diff2) * ao + 0.08 * ao;
    float specular = spec * shadow * 0.6;

    // Subsurface scatter hint (rim glow)
    float rim = pow(1.0 - max(dot(n, -rd), 0.0), 3.0) * 0.3;

    // Depth-based fog
    float fog = 1.0 - exp(-d * 0.08);
    vec3 fogCol = vec3(0.03, 0.01, 0.06);

    vec3 surface = vec3(lighting + specular + rim);
    surface = mix(surface, fogCol, fog);

    // Feedback ghost trail on the surface
    surface = mix(surface, prev_col, feedback_amount * 0.12);

    return vec3(d, surface.r, surface.g);
}

// Legacy 2D-compatible wrapper (returns scalar for non-SDF pipelines)
float field_sdf(vec2 p, float t,
                int primitive_type, float radius,
                float twist, int march_steps, float camera_z) {
    vec3 ro = vec3(0.0, 0.0, camera_z);
    vec3 rd = normalize(vec3(p, -1.5));
    float d = raymarch(ro, rd, t, primitive_type, radius, 0.0, 0.0, 0.0, 0.0,
                       march_steps, 0, 0.25);
    if (d > 25.0) {
        return 0.1 + 0.1 * sin(p.x * 3.0 + t) * sin(p.y * 3.0 + t * 0.7);
    }
    vec3 pos = ro + rd * d;
    vec3 n = calcNormal(pos, t, primitive_type, radius, 0.0, 0.0, 0.0, 0.0, 0, 0.25);
    vec3 light = normalize(vec3(0.5, 0.8, 0.6));
    float diff = max(dot(n, light), 0.0);
    float spec = pow(max(dot(reflect(-light, n), -rd), 0.0), 32.0);
    return diff * 0.7 + spec * 0.3;
}

#endif
