// Minimal binary-little-endian PLY reader for Pearlscape clouds.
//
// Handles exactly the format ply_export.py writes: a single `vertex` element
// with float/uchar properties (x y z, red green blue, and optionally field +
// dither). Returns plain typed arrays plus the bead_diameter / color_dither
// header comments. Replaces three's PLYLoader so we can read the extra
// per-bead properties it would otherwise drop.

const TYPES = {
  float: { size: 4, read: (dv, o) => dv.getFloat32(o, true) },
  float32: { size: 4, read: (dv, o) => dv.getFloat32(o, true) },
  double: { size: 8, read: (dv, o) => dv.getFloat64(o, true) },
  float64: { size: 8, read: (dv, o) => dv.getFloat64(o, true) },
  uchar: { size: 1, read: (dv, o) => dv.getUint8(o) },
  uint8: { size: 1, read: (dv, o) => dv.getUint8(o) },
  char: { size: 1, read: (dv, o) => dv.getInt8(o) },
  int8: { size: 1, read: (dv, o) => dv.getInt8(o) },
  ushort: { size: 2, read: (dv, o) => dv.getUint16(o, true) },
  uint16: { size: 2, read: (dv, o) => dv.getUint16(o, true) },
  short: { size: 2, read: (dv, o) => dv.getInt16(o, true) },
  int16: { size: 2, read: (dv, o) => dv.getInt16(o, true) },
  int: { size: 4, read: (dv, o) => dv.getInt32(o, true) },
  int32: { size: 4, read: (dv, o) => dv.getInt32(o, true) },
  uint: { size: 4, read: (dv, o) => dv.getUint32(o, true) },
  uint32: { size: 4, read: (dv, o) => dv.getUint32(o, true) },
};

export function parsePLY(arrayBuffer) {
  const bytes = new Uint8Array(arrayBuffer);
  const token = "end_header\n";
  // The header is ASCII; find its end without decoding the whole (binary) file.
  const headText = new TextDecoder("ascii").decode(
    bytes.subarray(0, Math.min(bytes.length, 65536))
  );
  const tokEnd = headText.indexOf(token);
  if (tokEnd < 0) throw new Error("not a PLY (no end_header)");
  const bodyOffset = tokEnd + token.length;

  let format = null;
  let count = 0;
  let inVertex = false;
  const props = []; // {name, size, read, offset}
  let stride = 0;
  let beadDiameter = null;
  let colorDither = null;

  for (const raw of headText.slice(0, tokEnd).split("\n")) {
    const line = raw.trim();
    if (!line) continue;
    const t = line.split(/\s+/);
    if (t[0] === "format") format = t[1];
    else if (t[0] === "comment" && t[1] === "bead_diameter") beadDiameter = parseFloat(t[2]);
    else if (t[0] === "comment" && t[1] === "color_dither") colorDither = parseFloat(t[2]);
    else if (t[0] === "element") {
      inVertex = t[1] === "vertex";
      if (inVertex) count = parseInt(t[2], 10);
    } else if (t[0] === "property" && inVertex) {
      const spec = TYPES[t[1]];
      if (!spec) throw new Error(`unsupported PLY property type: ${t[1]}`);
      props.push({ name: t[2], size: spec.size, read: spec.read, offset: stride });
      stride += spec.size;
    }
  }

  if (format !== "binary_little_endian") {
    throw new Error(`unsupported PLY format: ${format} (need binary_little_endian)`);
  }

  const dv = new DataView(arrayBuffer, bodyOffset);
  const find = (name) => props.find((p) => p.name === name);
  const has = (name) => find(name) !== undefined;

  const position = new Float32Array(count * 3);
  const color = has("red") ? new Uint8Array(count * 3) : null;
  const field = has("field") ? new Float32Array(count) : null;
  const dither = has("dither") ? new Float32Array(count) : null;

  const px = find("x"), py = find("y"), pz = find("z");
  const pr = find("red"), pg = find("green"), pb = find("blue");
  const pf = find("field"), pd = find("dither");

  for (let i = 0; i < count; i++) {
    const base = i * stride;
    position[i * 3] = px.read(dv, base + px.offset);
    position[i * 3 + 1] = py.read(dv, base + py.offset);
    position[i * 3 + 2] = pz.read(dv, base + pz.offset);
    if (color) {
      color[i * 3] = pr.read(dv, base + pr.offset);
      color[i * 3 + 1] = pg.read(dv, base + pg.offset);
      color[i * 3 + 2] = pb.read(dv, base + pb.offset);
    }
    if (field) field[i] = pf.read(dv, base + pf.offset);
    if (dither) dither[i] = pd.read(dv, base + pd.offset) / 255.0; // uchar -> [0,1]
  }

  return { count, position, color, field, dither, beadDiameter, colorDither };
}
