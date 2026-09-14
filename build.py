"""Pack the trained model into a QR code. 2953 bytes is the spec, not a target."""
import base64, gzip, json, pathlib, re, subprocess, sys
import numpy as np
import segno

QR_MAX = 2953                 # version 40, error correction L, byte mode

# Chrome blocks top-frame data: URI navigation outright, so the self-contained
# payload is Safari-only. Putting the same bytes in a URL *fragment* works in
# every browser and still never sends the model to a server: fragments are not
# transmitted in the HTTP request. The page at PAGE_URL is a static stub that
# only inflates what the scanner already carried.
PAGE_URL = "https://123satyajeet123.github.io/mnist-qr/#"

# Scanners need dark modules on a light field; inverting fails outright.
QR_DARK, QR_LIGHT = "#0e3a35", "#f5f8f7"
CONV_OUT, KERNEL, CLASSES = 24, 5, 10
HERE = pathlib.Path(__file__).parent


def quantize(values):
    """int8 + one float32 scale. The header is 26-50 numbers; fp32 wastes 3/4 of it."""
    step = float(np.abs(values).max()) / 127.0 or 1.0
    return np.clip(np.round(values / step), -127, 127).astype(np.int8), step


def pack(model):
    filters, grid = int(model["filters"]), CONV_OUT // int(model["pool"])
    qs, ks = quantize(model["scale"])
    qh, kh = quantize(model["shift"])
    qt, kt = quantize(model["t2"] / float(model["a2"]))
    header = bytes([filters, grid]) + np.array([ks, kh, kt], "<f4").tobytes()
    bits = np.concatenate([model["w1s"].ravel() > 0, model["w2s"].ravel() > 0])
    return header + qs.tobytes() + qh.tobytes() + qt.tobytes() + np.packbits(bits).tobytes()


def unpack(blob):
    """numpy mirror of U() in infer.js — proves the bytes decode to what we packed."""
    b = np.frombuffer(blob, np.uint8)
    F, G = int(b[0]), int(b[1])
    ks, kh, kt = np.frombuffer(blob[2:14], "<f4")
    i8 = lambda o, n: b[o:o + n].astype(np.int8).astype(np.float32)
    sc, sh = i8(14, F) * ks, i8(14 + F, F) * kh
    t2 = i8(14 + 2 * F, 10) * kt
    flat = np.unpackbits(b[14 + 2 * F + 10:])
    w1 = np.where(flat[:25 * F], 1.0, -1.0).reshape(KERNEL * KERNEL, F)
    off = 25 * F
    w2 = np.where(flat[off:off + G * G * F * 10], 1.0, -1.0).reshape(G * G * F, CLASSES)
    return F, G, sc, sh, t2, w1, w2


def predict(images, blob):
    """numpy mirror of P() — the accuracy this reports is what the QR will do."""
    from numpy.lib.stride_tricks import sliding_window_view
    F, G, sc, sh, t2, w1, w2 = unpack(blob)
    S = CONV_OUT // G
    patches = sliding_window_view(images, (KERNEL, KERNEL), axis=(1, 2))
    patches = np.ascontiguousarray(patches).reshape(len(images), CONV_OUT * CONV_OUT, -1)
    act = np.maximum(sc * (patches @ w1) + sh, 0).reshape(-1, CONV_OUT, CONV_OUT, F)
    pooled = act.reshape(-1, G, S, G, S, F).max(axis=(2, 4))
    return (pooled.reshape(len(images), -1) @ w2 + t2).argmax(1)


def page_html(blob):
    """Assemble the page from `page.src.html` and `infer.js`.

    The sources are written to be read; terser does the squeezing. Hand-minified
    source would save nothing here and cost everything in review.
    """
    source = (HERE / "page.src.html").read_text()
    script = re.search(r"<script>(.*)</script>", source, re.S).group(1)
    combined = (HERE / "infer.js").read_text() + "\n" + script.replace(
        "MODEL_BASE64", json.dumps(blob))

    minified = subprocess.run(
        ["npx", "--no-install", "terser", "--compress", "--mangle", "--toplevel"],
        input=combined, capture_output=True, text=True, check=True,
        cwd=HERE).stdout.strip()

    markup = source[:source.index("<script>")]
    markup = re.sub(r"/\*.*?\*/", "", markup, flags=re.S)        # CSS comments
    markup = re.sub(r"\s*\n\s*", "", markup)                      # line breaks
    markup = re.sub(r"\s*([{};:,])\s*", r"\1", markup)            # CSS padding
    return markup + "<script>" + minified + "</script>"


def viewer_html(blob, model):
    """The hosted page that wraps the scanned payload.

    It is served rather than scanned, so it costs the QR nothing. It unpacks the
    same weights from the same bytes, so it cannot drift from what is running.
    """
    source = (HERE / "viewer.src.html").read_text()
    # The splice point is a comment, so the source stays valid on its own.
    source = re.sub(r"/\* INFER_JS is spliced.*?\*/",
                    lambda _: (HERE / "infer.js").read_text(),
                    source, count=1, flags=re.S)
    source = source.replace("WEIGHT_BYTES", str(len(pack(model))))
    source = source.replace("ACCURACY", f"{float(model['full_acc']) * 100:.2f}")
    return source


def styled_qr(url, model, dest, digit=3, share=0.045):
    """A QR in the project's colours with a real MNIST digit in the middle.

    The art budget is the error-correction budget and nothing more. Measured on
    this payload (version 40-L): a 7% centre hole still decodes, 8% does not.
    Image-style QR art needs level H, which holds 1,273 bytes at version 40 —
    less than half of what is in here, so it is not an option at any setting.

    The mark is a test-set digit rather than a convolution kernel: a 5x5 binary
    kernel is honest but reads as a random box, which helps nobody.
    """
    from PIL import Image, ImageDraw

    qr = segno.make(url, error="l", mode="byte")
    qr.save(dest, scale=12, border=4, dark=QR_DARK, light=QR_LIGHT)

    # model.npz carries the first 300 test images, so this needs no dataset.
    labels, images = model["yte"], model["xte"]
    sample = images[int(np.where(labels == digit)[0][0])]

    canvas = Image.open(dest).convert("RGB")
    side = canvas.size[0]
    draw = ImageDraw.Draw(canvas)

    span = int(side * share ** 0.5)
    span -= span % 28                       # whole pixels per source pixel
    origin = (side - span) // 2
    pad = span // 9
    draw.rounded_rectangle(
        [origin - pad, origin - pad, origin + span + pad, origin + span + pad],
        radius=span // 6, fill=QR_LIGHT)

    cell = span // 28
    for row in range(28):
        for col in range(28):
            if sample[row, col] > 0.45:
                draw.rectangle([origin + col * cell, origin + row * cell,
                                origin + (col + 1) * cell - 1,
                                origin + (row + 1) * cell - 1], fill=QR_DARK)
    canvas.save(dest)
    return qr.version


def build(model_path):
    model = np.load(model_path)
    raw = pack(model)
    blob = base64.b64encode(raw).decode()
    page = page_html(blob)
    compressed = gzip.compress(page.encode(), 9, mtime=0)

    # The QR carries a URL whose fragment is the entire program. Fragments are
    # never sent to a server, so the model still travels inside the code; the
    # page it lands on only inflates what the scanner already had. base64url
    # keeps every character legal in a URL.
    fragment = base64.urlsafe_b64encode(compressed).decode().rstrip("=")
    url = PAGE_URL + fragment
    size = len(url.encode())

    images, labels = model["xte"], model["yte"]
    preds = predict(images, raw)
    packed_acc = float((preds == labels).mean())
    code_only = len(gzip.compress(page.replace(blob, "").encode(), 9))

    print(f"{model_path}: F={int(model['filters'])} pool={int(model['pool'])}")
    print(f"  weights {len(raw):>5} B   code(gzip) {code_only:>5} B   "
          f"page(gzip) {len(compressed):>5} B")
    print(f"  QR payload {size:>5} / {QR_MAX} B   "
          f"{'FITS +' + str(QR_MAX - size) if size <= QR_MAX else 'OVER by ' + str(size - QR_MAX)}")
    print(f"  float acc {float(model['acc']):.4f}  ->  packed acc {packed_acc:.4f} "
          f"(n={len(labels)})")

    # Written unconditionally: page.html is for reading, testset.json is what
    # verify.mjs checks against. Neither should depend on whether the QR fits.
    (HERE / "page.html").write_text(page)
    json.dump({"blob": blob, "labels": labels.tolist(), "ref": preds.tolist(),
               "px": base64.b64encode((images * 255).astype(np.uint8).tobytes()).decode()},
              open(HERE / "testset.json", "w"))

    if size > QR_MAX:
        return False

    docs = HERE / "docs"
    docs.mkdir(exist_ok=True)
    (HERE / "payload_url.txt").write_text(url)
    (docs / "index.html").write_text(viewer_html(blob, model))
    # A clickable equivalent of scanning, for anyone without a camera to hand.
    (docs / "demo.html").write_text(
        '<!doctype html><meta charset=utf-8><title>MNIST in a QR code</title>'
        f'<script>location.replace("./#" + "{fragment}")</script>')

    qr = segno.make(url, error="l", mode="byte")
    qr.save(HERE / "mnist_qr.png", scale=16, border=4)
    version = styled_qr(url, model, HERE / "mnist_qr_styled.png")
    print(f"  QR version {version}-L  ->  mnist_qr.png, mnist_qr_styled.png")
    return True


if __name__ == "__main__":
    ok = [build(p) for p in sys.argv[1:]]
    sys.exit(0 if any(ok) else 1)
