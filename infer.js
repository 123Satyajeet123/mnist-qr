// The model and the classifier. Spliced into the QR payload, the viewer and
// verify.mjs, so the three cannot drift.
//
// Names are long because terser mangles them at build time.

// Unpack the model from the bytes the QR code carries.
//
// Byte layout:
//   [0]          number of convolution filters
//   [1]          pooled grid size (the feature map is grid x grid after pooling)
//   [2 .. 14)    three float32 scales, one for each int8 array below
//   [14 .. )     int8 channel scale, int8 channel shift, int8 per-class bias
//   then         one bit per weight, most significant first:
//                first the convolution kernels, then the dense layer
function unpackModel(base64) {
  const raw = atob(base64);
  const bytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);

  const header = new DataView(bytes.buffer);
  const filters = bytes[0];
  const grid = bytes[1];

  // Each int8 array shares a single float32 scale. Undo that here.
  function dequantise(offset, count, scale) {
    const out = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      out[i] = (bytes[offset + i] << 24 >> 24) * scale;   // sign-extend to int8
    }
    return out;
  }

  const channelScale = dequantise(14, filters, header.getFloat32(2, true));
  const channelShift = dequantise(14 + filters, filters, header.getFloat32(6, true));
  const classBias = dequantise(14 + 2 * filters, 10, header.getFloat32(10, true));

  // Every weight is one bit, meaning +1 or -1.
  let bit = (14 + 2 * filters + 10) * 8;
  function readSigns(count) {
    const out = new Int8Array(count);
    for (let i = 0; i < count; i++, bit++) {
      out[i] = (bytes[bit >> 3] >> (7 - (bit & 7))) & 1 ? 1 : -1;
    }
    return out;
  }

  return {
    filters,
    grid,
    channelScale,
    channelShift,
    classBias,
    convWeights: readSigns(25 * filters),
    denseWeights: readSigns(grid * grid * filters * 10),
  };
}

// Score one image against all ten digits. `pixels` is a Float32Array of 784
// values in 0..1, row-major 28x28. The caller picks the argmax.
//
// 5x5 convolution -> BatchNorm -> ReLU -> max-pool -> dense.
function scoreDigits(pixels, model) {
  const { filters, grid, convWeights, denseWeights } = model;
  const poolSize = 24 / grid;

  // Convolution. No padding, so 28x28 becomes 24x24.
  const activations = new Float32Array(24 * 24 * filters);
  for (let y = 0; y < 24; y++) {
    for (let x = 0; x < 24; x++) {
      for (let f = 0; f < filters; f++) {
        let sum = 0;
        for (let tap = 0; tap < 25; tap++) {
          const pixel = (y + (tap / 5 | 0)) * 28 + x + tap % 5;
          sum += pixels[pixel] * convWeights[tap * filters + f];
        }
        // BatchNorm was folded into one scale and shift per channel at export.
        const value = model.channelScale[f] * sum + model.channelShift[f];
        activations[(y * 24 + x) * filters + f] = value > 0 ? value : 0;
      }
    }
  }

  // Max-pool and accumulate the dense layer in one pass.
  const logits = Float32Array.from(model.classBias);
  for (let row = 0; row < grid; row++) {
    for (let col = 0; col < grid; col++) {
      for (let f = 0; f < filters; f++) {
        let peak = 0;
        for (let dy = 0; dy < poolSize; dy++) {
          for (let dx = 0; dx < poolSize; dx++) {
            const at = ((row * poolSize + dy) * 24 + col * poolSize + dx) * filters + f;
            if (activations[at] > peak) peak = activations[at];
          }
        }
        const base = ((row * grid + col) * filters + f) * 10;
        for (let digit = 0; digit < 10; digit++) {
          logits[digit] += peak * denseWeights[base + digit];
        }
      }
    }
  }

  return logits;
}
