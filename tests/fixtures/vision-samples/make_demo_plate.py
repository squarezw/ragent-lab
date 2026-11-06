"""Regenerate demo_plate/image.png, the self-drawn sample used by tests and local dev.

Run from the repo root: python tests/fixtures/vision-samples/make_demo_plate.py
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PX = 4  # pixels per millimetre
OUT = Path(__file__).parent / 'demo_plate' / 'image.png'


def main() -> None:
    img = Image.new('RGB', (1100, 720), 'white')
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=22)
    small = ImageFont.load_default(size=18)

    def mm(x0, y0, x, y):
        return x0 + x * PX, y0 + y * PX

    def dashed(p, q, dash=12, gap=6, width=1):
        (x1, y1), (x2, y2) = p, q
        length = max(abs(x2 - x1), abs(y2 - y1))
        steps = int(length // (dash + gap)) + 1
        for i in range(steps):
            a = i * (dash + gap) / length
            b = min((i * (dash + gap) + dash) / length, 1)
            d.line([(x1 + (x2 - x1) * a, y1 + (y2 - y1) * a),
                    (x1 + (x2 - x1) * b, y1 + (y2 - y1) * b)], fill='black', width=width)

    def arrow_dim(p, q, text, offset_text):
        d.line([p, q], fill='black', width=1)
        for end, other in ((p, q), (q, p)):
            dx, dy = other[0] - end[0], other[1] - end[1]
            n = max(abs(dx), abs(dy)) or 1
            ux, uy = dx / n * 12, dy / n * 12
            d.polygon([end, (end[0] + ux - uy * 0.35, end[1] + uy + ux * 0.35),
                       (end[0] + ux + uy * 0.35, end[1] + uy - ux * 0.35)], fill='black')
        d.text(offset_text, text, fill='black', font=font, anchor='mm')

    def diameter(pos, prefix, number):
        # The bundled font has no Ø glyph, so draw the symbol as a circle with a slash.
        x, y = pos
        if prefix:
            d.text((x, y), prefix, fill='black', font=font, anchor='lm')
            x += d.textlength(prefix, font=font) + 2
        r = 7
        d.ellipse([x, y - r, x + 2 * r, y + r], outline='black', width=2)
        d.line([(x - 1, y + r + 2), (x + 2 * r + 1, y - r - 2)], fill='black', width=2)
        d.text((x + 2 * r + 4, y), number, fill='black', font=font, anchor='lm')

    # Front view: 120 x 60 plate, centre hole phi20, 2-phi8 at 15 from each end.
    fx, fy = 100, 140
    d.rectangle([mm(fx, fy, 0, 0), mm(fx, fy, 120, 60)], outline='black', width=3)
    cx, cy = mm(fx, fy, 60, 30)
    d.ellipse([cx - 10 * PX, cy - 10 * PX, cx + 10 * PX, cy + 10 * PX], outline='black', width=3)
    for hx in (15, 105):
        x, y = mm(fx, fy, hx, 30)
        d.ellipse([x - 4 * PX, y - 4 * PX, x + 4 * PX, y + 4 * PX], outline='black', width=3)
        dashed((x, y - 8 * PX), (x, y + 8 * PX), dash=18, gap=4)
    dashed(mm(fx, fy, -6, 30), mm(fx, fy, 126, 30), dash=18, gap=4)
    dashed((cx, cy - 14 * PX), (cx, cy + 14 * PX), dash=18, gap=4)

    # Front view dimensions.
    arrow_dim(mm(fx, fy, 0, -10), mm(fx, fy, 120, -10), '120', mm(fx, fy, 60, -15))
    arrow_dim(mm(fx, fy, -12, 0), mm(fx, fy, -12, 60), '60', mm(fx, fy, -19, 30))
    arrow_dim(mm(fx, fy, 15, 70), mm(fx, fy, 105, 70), '90', mm(fx, fy, 60, 75))
    diameter(mm(fx, fy, 74, 14), '', '20')
    diameter(mm(fx, fy, 4, 46), '2-', '8')

    # Side view (to the right): thickness 10, hidden lines for the holes.
    sx, sy = 700, 140
    d.rectangle([mm(sx, sy, 0, 0), mm(sx, sy, 10, 60)], outline='black', width=3)
    for top, bottom in ((20, 40), (26, 34)):
        dashed(mm(sx, sy, 0, top), mm(sx, sy, 10, top))
        dashed(mm(sx, sy, 0, bottom), mm(sx, sy, 10, bottom))
    dashed(mm(sx, sy, -5, 30), mm(sx, sy, 15, 30), dash=18, gap=4)
    arrow_dim(mm(sx, sy, 0, -10), mm(sx, sy, 10, -10), '10', mm(sx, sy, 5, -15))

    # Title block.
    tx, ty = 760, 560
    d.rectangle([tx, ty, tx + 300, ty + 130], outline='black', width=2)
    for i, line in enumerate(['DEMO PLATE', 'DWG NO. DEMO-01', 'MATERIAL Q235',
                              'SCALE 1:1  UNIT mm']):
        d.text((tx + 14, ty + 14 + i * 28), line, fill='black', font=small)
    d.text((100, 600), 'NOTE: BREAK ALL SHARP EDGES.', fill='black', font=small)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT, optimize=True)


if __name__ == '__main__':
    main()
