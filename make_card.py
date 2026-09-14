"""Render the post image.

Square, because the code has to survive X's downscale. A phone camera needs
roughly four pixels per QR module; this symbol is 185 modules wide including its
quiet zone, so it must land at 740px or more on the reader's screen. At 16:9
there is no room for both that and type.

The palette is the code's own two colours, so the symbol sits on the page
instead of floating on it.
"""
from PIL import Image, ImageDraw, ImageFont

import build

SIZE = 1500
QR_SIDE = 1080          # 5.8 px per module here, ~4.7 after X resizes to 1200
MARGIN = 120

PAPER = build.QR_LIGHT   # "#f5f8f7"
DEEP = build.QR_DARK     # "#0e3a35"
MUTED = "#6c807b"
FAINT = "#9aaaa5"

SF = "/System/Library/Fonts/SFNS.ttf"
MONO = "/System/Library/Fonts/SFNSMono.ttf"

TYPE = {
    "display": (SF, 60, "Semibold"),
    "body": (SF, 27, "Regular"),
    "data": (MONO, 23, None),
}


def font(role):
    path, size, weight = TYPE[role]
    face = ImageFont.truetype(path, size)
    if weight:
        face.set_variation_by_name(weight)
    return face


def run(draw, xy, parts, face):
    """Draw one line built from (text, colour) pairs."""
    x, y = xy
    for text, colour in parts:
        draw.text((x, y), text, font=face, fill=colour)
        x += draw.textlength(text, font=face)


def main():
    card = Image.new("RGB", (SIZE, SIZE), PAPER)
    draw = ImageDraw.Draw(card)

    qr = Image.open("mnist_qr_styled.png").convert("RGB").resize(
        (QR_SIDE, QR_SIDE), Image.LANCZOS)
    card.paste(qr, ((SIZE - QR_SIDE) // 2, 92))

    text_top = 92 + QR_SIDE + 74
    draw.text((MARGIN, text_top), "This QR code is a neural network.",
              font=font("display"), fill=DEEP)
    draw.text((MARGIN, text_top + 82), "Scan it and draw a digit.",
              font=font("body"), fill=MUTED)

    run(draw, (MARGIN, text_top + 142), [
        ("627 B", DEEP), (" of weights", MUTED),
        ("   2,948", DEEP), (" / 2,953 bytes", MUTED),
        ("   96.37%", DEEP), (" on MNIST", MUTED),
    ], font("data"))

    card.save("x_card.png")
    print(f"  x_card.png  {SIZE}x{SIZE}, QR {QR_SIDE}px, "
          f"{QR_SIDE / 185:.1f} px per module "
          f"({QR_SIDE / 185 * 1200 / SIZE:.1f} after X resizes to 1200)")


if __name__ == "__main__":
    main()
