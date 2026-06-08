// Glass-bead point-sprite shaders, mirroring the Rhino sprite look
// (display.py::_make_bead_sprites): a colour-tinted body with a bottom-left
// colour glint and a sharp white top-right specular, on a soft-edged disc.
//
// Colour is computed in the vertex shader from a per-bead field value quantized
// against a palette uniform — so switching palette is a uniform update, with no
// CPU recolour of the points. This mirrors color.py::quantize exactly:
//     t = field * m + dither * (ditherRand - 0.5);  idx = clamp(floor(t), 0, m-1)
// Beads from a plain PLY (no field) fall back to the baked `color` attribute.

export const MAX_PALETTE = 16; // must match the array size below

export const vertexShader = /* glsl */ `
  attribute vec3 color;          // baked colour (fallback when useField == false)
  attribute float field;         // [0,1) colour-field value
  attribute float ditherRand;    // [0,1) dither source
  uniform float beadDiameter;    // mm (world units)
  uniform float sizeScale;       // drawingBufferHeight / (2 * tan(fov/2))
  uniform vec3 palette[${MAX_PALETTE}];
  uniform int paletteSize;
  uniform float ditherAmt;
  uniform bool useField;
  varying vec3 vColor;

  void main() {
    if (useField) {
      float m = float(paletteSize);
      float t = field * m + ditherAmt * (ditherRand - 0.5);
      int idx = int(clamp(floor(t), 0.0, m - 1.0));
      vColor = palette[idx];     // dynamic uniform-array index (OK in vertex stage)
    } else {
      vColor = color;
    }
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    // World-sized sprite: a bead of diameter D at depth d projects to
    // D * sizeScale / d framebuffer pixels, so beads shrink with distance.
    gl_PointSize = clamp(beadDiameter * sizeScale / -mvPosition.z, 1.0, 4096.0);
    gl_Position = projectionMatrix * mvPosition;
  }
`;

export const fragmentShader = /* glsl */ `
  precision highp float;
  varying vec3 vColor;

  void main() {
    // Centered sprite coords in [-1, 1]; y points down (gl_PointCoord origin is
    // top-left), matching the image-space layout the Rhino bitmaps were built in.
    vec2 c = (gl_PointCoord - 0.5) * 2.0;
    float t = length(c);                       // 0 centre .. 1 rim
    float aa = 0.03;                           // ~2px antialiased edge at 64px
    float disc = clamp((1.0 - t) / aa, 0.0, 1.0);
    if (disc <= 0.0) discard;

    // Body luminance: brightens outward from a point nudged toward top-right.
    float bt = clamp(length(c - vec2(0.08, -0.08)), 0.0, 1.0);
    float lum = mix(0.45, 0.88, bt);

    // Colour glint, bottom-left.
    float gd = length(c - vec2(-0.40, 0.40));
    lum = clamp(lum + 0.6 * pow(clamp(1.0 - gd / 0.32, 0.0, 1.0), 2.0), 0.0, 1.0);

    vec3 body = vColor * lum;                   // multiply tint (Rhino body pass)

    // Sharp white specular, top-right.
    float sd = length(c - vec2(0.40, -0.40));
    float spec = clamp((0.15 - sd) / aa, 0.0, 1.0);

    vec3 col = mix(body, vec3(1.0), spec);      // white specular over the body
    gl_FragColor = vec4(col, disc);
  }
`;
