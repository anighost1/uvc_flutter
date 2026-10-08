#include <flutter/runtime_effect.glsl>

uniform vec2 uSize;
uniform float uYaw;
uniform float uPitch;
uniform float uFov;
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

  float lon = atan(d.x, d.z);
  float lat = asin(clamp(d.y, -1.0, 1.0));
  vec2 uv = vec2(lon / 6.2831853 + 0.5, 0.5 - lat / 3.1415926);
  fragColor = texture(uTex, uv);
}
