# MNIST in a QR code

A handwritten-digit classifier — weights, inference and interface — inside one
QR code. Scan it, draw a digit, it answers. Nothing is fetched.

![the classifier](mnist_qr_styled.png)

**[Try it →](https://123satyajeet123.github.io/mnist-qr/demo.html)**

**96.37%** on the full MNIST test set · **627 B** of weights, one bit each ·
**2,948 / 2,953 bytes** — version 40 is the last in the QR spec, so that is the
ceiling, not a target.

## How

```
https://123satyajeet123.github.io/mnist-qr/#H4sIAAAAAAAC_41Xa…
└───────────────── 44 B ─────────────────┘└────── 2,904 B ──────┘
                                           gzip(page), base64url
```

URL fragments are never sent to a server, so the model travels in the code you
scanned. The page inflates it and runs it, and holds no model of its own.

A pure `data:` URI would need no page at all, but Chrome blocks top-frame
`data:` navigation, so it would be Safari-only.

## Architecture

```
28×28 ──conv 5×5 ×24──> 24×24 ──ReLU──> ──maxpool 6×6──> 4×4 ──dense──> 10
           600 bits                                          3,840 bits
```

Every weight is one bit, +1 or −1. Plus 72 B of per-channel scales, shifts and
biases = 627 B.

The dense layer costs `grid² × filters × 10` bits, so pooling harder is what
buys filters. That is why the pool is 6×6 and not 2×2.

## Build

```sh
python3 -m venv --system-site-packages .venv && .venv/bin/pip install segno
npm install terser

.venv/bin/python train.py mnist_data 24 6 model.npz   # downloads MNIST, ~15 min
.venv/bin/python build.py model.npz                   # writes docs/ and the QR
node verify.mjs                                       # gate: JS must match numpy
```

## Files

| | |
|---|---|
| `train.py` | binary-weight CNN, explicit gradients, numpy only |
| `infer.js` | unpack + score, spliced into the payload, the viewer and the test |
| `page.src.html` | what the QR carries; terser minifies it at build time |
| `viewer.src.html` | the page it lands on |
| `build.py` | packs the bits, assembles both pages, emits the QR |
| `verify.mjs` | the gate: JS predictions must match numpy on 300 images |
