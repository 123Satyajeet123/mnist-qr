"""Pack the trained model into a QR code. 2953 bytes is the spec, not a target."""
import base64, gzip, json, pathlib, re, sys
import numpy as np
import segno

QR_MAX = 2953                 # version 40, error correction L, byte mode

# Chrome blocks top-frame data: URI navigation outright, so the self-contained
# payload is Safari-only. Putting the same bytes in a URL *fragment* works in
# every browser and still never sends the model to a server: fragments are not
# transmitted in the HTTP request. The page at PAGE_URL is a static stub that
# only inflates what the scanner already carried.
PAGE_URL = "https://123satyajeet123.github.io/mnist-qr/#"
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
    """MNIST normalises each digit into a 20x20 box centred in 28x28. drawImage
    does that resample natively, so the UI does no pixel arithmetic of its own."""
    infer = re.sub(r"\s*\n\s*", "", re.sub(r"^\s*//.*$", "", (HERE / "infer.js").read_text(), flags=re.M))
    return (
        '<meta name=viewport content=width=device-width,initial-scale=1>'
        '<body style="margin:0;background:#111;color:#eee;font:16px system-ui;text-align:center">'
        '<canvas id=c width=280 height=280 style="background:#000;margin:14px;'
        'touch-action:none"></canvas><canvas id=k width=28 height=28 hidden></canvas>'
        '<div id=o style="font:700 64px system-ui;height:70px"></div>'
        f'<script>{infer}'
        f'var M=U("{blob}"),g=c.getContext("2d"),q=k.getContext("2d"),d=0;'
        'g.lineWidth=24;g.lineCap=g.lineJoin="round";g.strokeStyle="#fff";'
        'function m(e){var r=c.getBoundingClientRect();'
        'g.lineTo((e.clientX-r.left)*280/r.width,(e.clientY-r.top)*280/r.height)}'
        'c.onpointerdown=function(e){if(o.textContent){g.clearRect(0,0,280,280);o.textContent=""}'
        'd=1;g.beginPath();m(e)};'
        'c.onpointermove=function(e){if(d){m(e);g.stroke()}};'
        'c.onpointerup=function(){if(!d)return;d=0;'
        'var s=g.getImageData(0,0,280,280).data,a=280,b=280,u=0,w=0,i,X,Y,p=new Float32Array(784);'
        'for(i=3;i<313600;i+=4)if(s[i]){X=i/4|0;Y=X/280|0;X%=280;'
        'if(X<a)a=X;if(X>u)u=X;if(Y<b)b=Y;if(Y>w)w=Y}'
        'if(u<a)return;'
        'var W=u-a+1,H=w-b+1,z=20/(W>H?W:H);k.width=28;'
        'q.drawImage(c,a,b,W,H,(28-W*z)/2,(28-H*z)/2,W*z,H*z);'
        's=q.getImageData(0,0,28,28).data;'
        'for(i=0;i<784;i++)p[i]=s[i*4+3]/255;'
        'o.textContent=P(p,M)};'
        '</script>')


def build(model_path):
    model = np.load(model_path)
    raw = pack(model)
    blob = base64.b64encode(raw).decode()
    page = page_html(blob)
    gz = gzip.compress(page.encode(), 9, mtime=0)
    uri = ("data:text/html,<script>fetch('data:;base64," + base64.b64encode(gz).decode()
           + "').then(r=>new Response(r.body.pipeThrough(new DecompressionStream('gzip')))"
             ".text()).then(t=>document.write(t))</script>")
    size = len(uri.encode())

    images, labels = model["xte"], model["yte"]
    preds = predict(images, raw)
    acc = float((preds == labels).mean())

    code_gz = len(gzip.compress(page.replace(blob, "").encode(), 9))
    print(f"{model_path}: F={int(model['filters'])} pool={int(model['pool'])}")
    print(f"  weights {len(raw):>5} B   code(gzip) {code_gz:>5} B   page(gzip) {len(gz):>5} B")
    print(f"  QR payload {size:>5} / {QR_MAX} B   "
          f"{'FITS +' + str(QR_MAX - size) if size <= QR_MAX else 'OVER by ' + str(size - QR_MAX)}")
    print(f"  float acc {float(model['acc']):.4f}  ->  packed acc {acc:.4f} (n={len(labels)})")

    # Universal variant: same gzip stream, carried in a fragment instead.
    # base64url so no character in the payload is reserved in a URL.
    frag = base64.urlsafe_b64encode(gz).decode().rstrip("=")
    url = PAGE_URL + frag
    url_size = len(url.encode())
    print(f"  universal  {url_size:>5} / {QR_MAX} B   "
          f"{'FITS +' + str(QR_MAX - url_size) if url_size <= QR_MAX else 'OVER by ' + str(url_size - QR_MAX)}"
          f"   (works in every browser)")
    if url_size <= QR_MAX:
        (HERE / "payload_url.txt").write_text(url)
        # A clickable equivalent of scanning, for people without a camera handy.
        docs = HERE / "docs"
        docs.mkdir(exist_ok=True)
        (docs / "demo.html").write_text(
            '<!doctype html><meta charset=utf-8><title>MNIST in a QR code</title>'
            f'<script>location.replace("./#" + {frag!r})</script>'.replace("'", '"', 2))
        qr_url = segno.make(url, error="l", mode="byte")
        qr_url.save(HERE / "mnist_qr_universal.png", scale=16, border=4)
        print(f"  QR version {qr_url.version}-L  ->  mnist_qr_universal.png")

    if size <= QR_MAX:
        (HERE / "payload.txt").write_text(uri)
        (HERE / "page.html").write_text(page)
        json.dump({"blob": blob, "labels": labels.tolist(), "ref": preds.tolist(),
                   "px": base64.b64encode((images * 255).astype(np.uint8).tobytes()).decode()},
                  open(HERE / "testset.json", "w"))
        qr = segno.make(uri, error="l", mode="byte")
        qr.save(HERE / "mnist_qr.png", scale=16, border=4)  # 177 modules needs the pixels
        print(f"  QR version {qr.version}-L  ->  mnist_qr.png")
    return size <= QR_MAX


if __name__ == "__main__":
    ok = [build(p) for p in sys.argv[1:]]
    sys.exit(0 if any(ok) else 1)
