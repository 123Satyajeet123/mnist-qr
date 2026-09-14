"""Render the hero image: the network actually running, on real data.

No scan requirement here, so the frame is free. The QR goes in a second image
where it can be large enough to read.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1600, 900

BG = (13, 15, 17)
INK = "#eef2f4"
MUTED = "#79858c"
FAINT = "#3c464c"
LIVE = (78, 201, 165)

SF = "/System/Library/Fonts/SFNS.ttf"
MONO = "/System/Library/Fonts/SFNSMono.ttf"


def sf(size, weight="Regular"):
    face = ImageFont.truetype(SF, size)
    face.set_variation_by_name(weight)
    return face


def mono(size):
    return ImageFont.truetype(MONO, size)


def tint(level, cold=(26, 31, 34)):
    """Blend the accent over the panel colour by activation strength."""
    return tuple(int(cold[i] + (LIVE[i] - cold[i]) * level) for i in range(3))


def grey(value, side, box):
    """One channel array to an upscaled RGB tile."""
    tile = Image.fromarray((value * 255).astype(np.uint8), "L").convert("RGB")
    return tile.resize((box, box), Image.NEAREST)


def label(draw, xy, text):
    draw.text(xy, text, font=mono(17), fill=MUTED)


def main():
    state = np.load("/tmp/demo_state.npz")
    img, acts, logits = state["img"], state["acts"], state["logits"]
    filters = acts.shape[2]

    card = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(card)

    draw.text((88, 58), "627 bytes of neural network", font=sf(48, "Semibold"), fill=INK)
    draw.text((88, 122), "unpacked from a QR code, running on a digit you draw",
              font=sf(24), fill=MUTED)

    top, art = 212, 178

    # 1 - the input the network is handed.
    x1, side = 88, 300
    label(draw, (x1, top - 30), "INPUT  28×28")
    card.paste(grey(img, 28, side), (x1, top))

    # 2 - every filter's response, each at its own scale so quiet ones show.
    x2, cell, gap, cols = 470, 120, 10, 6
    label(draw, (x2, top - 30), f"CONV  {filters} filters, 5×5, one bit per weight")
    for f in range(filters):
        plane = acts[:, :, f]
        level = plane / (plane.max() or 1.0)
        tile = Image.new("RGB", (24, 24))
        tile.putdata([tint(v) for v in level.reshape(-1)])
        col, row = f % cols, f // cols
        card.paste(tile.resize((cell, cell), Image.NEAREST),
                   (x2 + col * (cell + gap), top + row * (cell + gap)))

    # 3 - the ten scores.
    x3 = x2 + cols * (cell + gap) + 78
    label(draw, (x3, top - 30), "SCORES")
    low, span = logits.min(), (logits.max() - logits.min()) or 1.0
    winner = int(logits.argmax())
    row_h, bar_w = 50, 190
    for digit in range(10):
        y = top + 4 + digit * row_h
        draw.text((x3, y - 3), str(digit), font=mono(21),
                  fill=LIVE if digit == winner else FAINT)
        width = max(4, int((logits[digit] - low) / span * bar_w))
        draw.rounded_rectangle([x3 + 34, y, x3 + 34 + width, y + 17], radius=4,
                               fill=LIVE if digit == winner else (45, 52, 56))

    draw.text((88, H - 92), "96.37% on MNIST   ·   2,948 of a QR code's 2,953 bytes"
              "   ·   github.com/123Satyajeet123/mnist-qr", font=mono(20), fill=FAINT)

    card.save("x_hero.png")
    print(f"  x_hero.png  {W}x{H}")


if __name__ == "__main__":
    main()
