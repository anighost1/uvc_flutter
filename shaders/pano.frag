#include <flutter/runtime_effect.glsl>

uniform vec2 uSize;
uniform float uYaw;
uniform float uPitch;
uniform float uFov;
uniform float uMode; // 0 = equirectangular, 1 = dual fisheye (side by side)
uniform float uLens; // fisheye lens field of view in radians (~3.49 = 200 deg)
uniform sampler2D uTex;

out vec4 fragColor;

void main() {
  vec2 p = (FlutterFragCoord().xy / uSize) * 2.0 - 1.0;
  p.x *= uSize.x / uSize.y;
  float f = 1.0 / tan(uFov * 0.5);
  vec3 d = normalize(vec3(p.x, -p.y, f));

  float cp = cos(uPitch), sp = sin(uPitch);
  d = vec3(d.x, d.y * cp - d.z * sp, d.y * sp + d.z * cp);
  float cy = cos(uYaw), sy = sin(uYaw);
  d = vec3(d.x * cy + d.z * sy, d.y, -d.x * sy + d.z * cy);

  vec2 uv;
  if (uMode < 0.5) {
    float lon = atan(d.x, d.z);
    float lat = asin(clamp(d.y, -1.0, 1.0));
    uv = vec2(lon / 6.2831853 + 0.5, 0.5 - lat / 3.1415926);
  } else {
    bool back = d.z < 0.0;
    vec3 e = back ? vec3(-d.x, d.y, -d.z) : d;
    float theta = acos(clamp(e.z, -1.0, 1.0));
    float r = theta / (uLens * 0.5);
    float len = length(e.xy);
    vec2 dir = len > 0.00001 ? e.xy / len : vec2(0.0);
    vec2 c = vec2(back ? 0.75 : 0.25, 0.5);
    uv = c + vec2(dir.x * r * 0.25, -dir.y * r * 0.5);
  }
  fragColor = texture(uTex, uv);
}
