// Inlined verbatim into the QR payload and loaded by verify.mjs, so the demo
// and its test can never drift. No module syntax.
function U(s) {
  const r = atob(s), b = new Uint8Array(r.length);
  for (let i = 0; i < r.length; i++) b[i] = r.charCodeAt(i);
  const v = new DataView(b.buffer), F = b[0], G = b[1];
  const q = (o, n, k) => { const a = new Float32Array(n); for (let i = 0; i < n; i++) a[i] = (b[o + i] << 24 >> 24) * k; return a; };
  const sc = q(14, F, v.getFloat32(2, 1)), sh = q(14 + F, F, v.getFloat32(6, 1)), t2 = q(14 + 2 * F, 10, v.getFloat32(10, 1));
  let bit = (14 + 2 * F + 10) * 8;
  const bits = n => { const a = new Int8Array(n); for (let i = 0; i < n; i++, bit++) a[i] = (b[bit >> 3] >> (7 - (bit & 7)) & 1) ? 1 : -1; return a; };
  return { F, G, sc, sh, t2, w1: bits(25 * F), w2: bits(G * G * F * 10) };
}

// px: Float32Array(784), 0..1, row-major 28x28.
function P(px, M) {
  const F = M.F, G = M.G, S = 24 / G, A = new Float32Array(24 * 24 * F);
  for (let y = 0; y < 24; y++) for (let x = 0; x < 24; x++) for (let f = 0; f < F; f++) {
    let d = 0;
    for (let k = 0; k < 25; k++) d += px[(y + (k / 5 | 0)) * 28 + x + k % 5] * M.w1[k * F + f];
    d = M.sc[f] * d + M.sh[f];
    A[(y * 24 + x) * F + f] = d > 0 ? d : 0;
  }
  const L = Float32Array.from(M.t2);
  for (let r = 0; r < G; r++) for (let c = 0; c < G; c++) for (let f = 0; f < F; f++) {
    let m = 0;
    for (let i = 0; i < S; i++) for (let j = 0; j < S; j++) { const v = A[((r * S + i) * 24 + c * S + j) * F + f]; if (v > m) m = v; }
    const o = ((r * G + c) * F + f) * 10;
    for (let k = 0; k < 10; k++) L[k] += m * M.w2[o + k];
  }
  let z = 0;
  for (let k = 1; k < 10; k++) if (L[k] > L[z]) z = k;
  return z;
}
