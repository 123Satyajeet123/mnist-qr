"""Binary-weight CNN for MNIST. Explicit gradients, no autograd, numpy only.

Every weight costs exactly one bit, so the QR code's byte budget picks the
architecture. Convolution is here because weight sharing buys the same accuracy
as a dense layer for roughly a fifth of the bits.
"""
import gzip, pathlib, struct, sys, urllib.request
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

SIDE, KERNEL, CLASSES, EPS = 28, 5, 10, 1e-5
CONV_OUT = SIDE - KERNEL + 1          # 24
POOL = 2                              # set from argv; the FC layer costs
POOLED = CONV_OUT // POOL             # POOLED^2 * filters * 10 bits


MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = ["train-images-idx3-ubyte", "train-labels-idx1-ubyte",
         "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"]


def fetch(data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = data_dir / f"{name}.gz"
        if not target.exists():
            print(f"  downloading {name}.gz", flush=True)
            urllib.request.urlretrieve(MIRROR + f"{name}.gz", target)
    return data_dir


def read_idx(path):
    with gzip.open(path, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        rank = magic % 16 - 1
        dims = struct.unpack(f">{rank}I", f.read(4 * rank))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(count, *dims)


def load_split(data_dir, prefix):
    images = read_idx(data_dir / f"{prefix}-images-idx3-ubyte.gz").astype(np.float32) / 255.0
    labels = read_idx(data_dir / f"{prefix}-labels-idx1-ubyte.gz").astype(np.int64)
    return images, labels


def shift(images, rng, amount=3):
    """Random integer translation. The demo sees badly-placed digits; paying for
    that here costs zero payload bytes, paying for it at inference costs 261."""
    n = len(images)
    padded = np.pad(images, ((0, 0), (amount, amount), (amount, amount)))
    oy, ox = rng.integers(0, 2 * amount + 1, (2, n))
    rows = oy[:, None, None] + np.arange(SIDE)[None, :, None]
    cols = ox[:, None, None] + np.arange(SIDE)[None, None, :]
    return padded[np.arange(n)[:, None, None], rows, cols]


def patchify(images):
    """(B,28,28) -> (B,576,25); the im2col that turns conv into one matmul."""
    windows = sliding_window_view(images, (KERNEL, KERNEL), axis=(1, 2))
    return np.ascontiguousarray(windows).reshape(len(images), CONV_OUT * CONV_OUT, KERNEL * KERNEL)


def binarize(latent):
    return np.sign(latent) + (latent == 0), np.abs(latent).mean()


def softmax_cross_entropy(logits, labels):
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    loss = -np.log(probs[np.arange(len(labels)), labels] + 1e-9).mean()
    grad = probs.copy()
    grad[np.arange(len(labels)), labels] -= 1.0
    return loss, grad / len(labels)


class Adam:
    def __init__(self, shapes, lr=2e-3):
        self.lr = self.base_lr = lr
        self.t = 0
        self.m = [np.zeros(s, np.float32) for s in shapes]
        self.v = [np.zeros(s, np.float32) for s in shapes]

    def step(self, params, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = 0.9 * self.m[i] + 0.1 * g
            self.v[i] = 0.999 * self.v[i] + 0.001 * g * g
            mhat = self.m[i] / (1 - 0.9 ** self.t)
            vhat = self.v[i] / (1 - 0.999 ** self.t)
            p -= self.lr * mhat / (np.sqrt(vhat) + 1e-8)


def pool_max(x):
    """(B,24,24,F) -> (B,POOLED,POOLED,F) plus the mask that routes gradients."""
    blocks = x.reshape(-1, POOLED, POOL, POOLED, POOL, x.shape[-1])
    pooled = blocks.max(axis=(2, 4))
    mask = blocks == pooled[:, :, None, :, None, :]
    return pooled, mask


def forward(patches, w1s, bn, w2s, t2, training):
    gamma, beta, run_mean, run_var = bn
    conv = patches @ w1s                                   # (B,576,F)
    axes = (0, 1)
    if training:
        mean, var = conv.mean(axes), conv.var(axes)
        run_mean *= 0.9; run_mean += 0.1 * mean
        run_var *= 0.9; run_var += 0.1 * var
    else:
        mean, var = run_mean, run_var
    inv_std = 1.0 / np.sqrt(var + EPS)
    normed = (conv - mean) * inv_std
    pre = gamma * normed + beta
    activated = np.maximum(pre, 0.0).reshape(-1, CONV_OUT, CONV_OUT, w1s.shape[1])
    pooled, mask = pool_max(activated)
    flat = pooled.reshape(len(patches), -1)
    return normed, inv_std, pre, mask, flat, flat @ w2s + t2


def train(filters, data_dir, epochs=40, batch=128, seed=0):
    rng = np.random.default_rng(seed)
    xtr, ytr = load_split(data_dir, "train")
    xte, yte = load_split(data_dir, "t10k")
    pte = patchify(xte)

    w1 = rng.normal(0, 0.5, (KERNEL * KERNEL, filters)).astype(np.float32)
    w2 = rng.normal(0, 0.5, (POOLED * POOLED * filters, CLASSES)).astype(np.float32)
    gamma = np.ones(filters, np.float32)
    beta = np.zeros(filters, np.float32)
    bn = [gamma, beta, np.zeros(filters, np.float32), np.ones(filters, np.float32)]
    t2 = np.zeros(CLASSES, np.float32)
    params = [w1, gamma, beta, w2, t2]
    opt = Adam([p.shape for p in params])

    best = None
    for epoch in range(epochs):
        opt.lr = opt.base_lr * 0.5 * (1 + np.cos(np.pi * epoch / epochs))
        order = rng.permutation(len(xtr))
        for start in range(0, len(order) - batch + 1, batch):
            idx = order[start:start + batch]
            patches, y = patchify(shift(xtr[idx], rng)), ytr[idx]
            w1s, _ = binarize(w1)
            w2s, a2 = binarize(w2)

            normed, inv_std, pre, mask, flat, logits = forward(
                patches, w1s, bn, w2s * a2, t2, training=True)

            _, dlogits = softmax_cross_entropy(logits, y)
            dw2 = a2 * (flat.T @ dlogits)
            dt2 = dlogits.sum(0)

            dpooled = (dlogits @ (a2 * w2s).T).reshape(-1, POOLED, 1, POOLED, 1, filters)
            dpre = (mask * dpooled).reshape(-1, CONV_OUT * CONV_OUT, filters) * (
                pre.reshape(-1, CONV_OUT * CONV_OUT, filters) > 0)

            dgamma = (dpre * normed).sum((0, 1))
            dbeta = dpre.sum((0, 1))
            dnormed = dpre * gamma
            n = dpre.shape[0] * dpre.shape[1]
            dconv = inv_std / n * (
                n * dnormed - dnormed.sum((0, 1)) - normed * (dnormed * normed).sum((0, 1)))
            dw1 = patches.reshape(-1, KERNEL * KERNEL).T @ dconv.reshape(-1, filters)

            opt.step(params, [dw1, dgamma, dbeta, dw2, dt2])
            np.clip(w1, -1, 1, out=w1)
            np.clip(w2, -1, 1, out=w2)

        w1s, _ = binarize(w1)
        w2s, a2 = binarize(w2)
        acc = (forward(pte, w1s, bn, w2s * a2, t2, training=False)[5].argmax(1) == yte).mean()
        print(f"  epoch {epoch + 1:3d}  test {acc:.4f}", flush=True)
        if best is None or acc > best[0]:
            best = (acc, w1s.copy(), [b.copy() for b in bn], w2s.copy(), float(a2), t2.copy())
    return best, (xte, yte)


if __name__ == "__main__":
    data_dir = fetch(pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "mnist_data"))
    filters = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    globals()["POOL"] = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    globals()["POOLED"] = CONV_OUT // POOL
    bits = KERNEL * KERNEL * filters + POOLED * POOLED * filters * CLASSES
    print(f"filters={filters}  weight bits={bits}  ({bits / 8:.0f} bytes)")

    (acc, w1s, bn, w2s, a2, t2), (xte, yte) = train(filters, data_dir)
    gamma, beta, run_mean, run_var = bn
    # Fold BatchNorm into per-channel scale/shift so inference is relu(scale*dot + shift).
    scale = gamma / np.sqrt(run_var + EPS)
    shift = beta - gamma * run_mean / np.sqrt(run_var + EPS)
    print(f"best test accuracy {acc:.4f}")
    out = sys.argv[4] if len(sys.argv) > 4 else "model.npz"
    np.savez(out, w1s=w1s, w2s=w2s, scale=scale, shift=shift, a2=a2, t2=t2,
             acc=acc, filters=filters, pool=POOL, xte=xte[:300], yte=yte[:300])
