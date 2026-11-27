"""Hand-drawn style architecture sketch behind README 'How it works' (docs/architecture.png).

Render with the Kalam font (OFL, google/fonts ofl/kalam) visible to fontconfig; rsvg-convert falls
back per glyph for the arrows and Greek letters Kalam lacks, resvg would swap the whole line:

    python3 docs/architecture_sketch.py /tmp/arch.svg
    rsvg-convert -b white /tmp/arch.svg -o docs/architecture.png
"""

import math
import random
import sys
from html import escape

random.seed(7)
W, H = 1870, 1290
out: list[str] = []
INK = "#1e1e1e"
FILL = {
    "yellow": "#fff3bf",
    "green": "#d3f9d8",
    "blue": "#d0ebff",
    "orange": "#ffd8a8",
    "pink": "#fcc2d7",
    "purple": "#e5dbff",
    "white": "#ffffff",
}
ORANGE = "#e8590c"
FONT = "Kalam"
SYMBOLS = "→αβσ⊙λ₀"  # glyphs Kalam lacks; drawn in DejaVu Sans


def j(a: float = 1.6) -> float:
    return random.uniform(-a, a)


def rough_line(x1, y1, x2, y2, stroke=INK, sw=1.8, dash=None, passes=2):
    for _ in range(passes):
        mx, my = (x1 + x2) / 2 + j(2.5), (y1 + y2) / 2 + j(2.5)
        d = f"M{x1+j():.1f},{y1+j():.1f} Q{mx:.1f},{my:.1f} {x2+j():.1f},{y2+j():.1f}"
        da = f' stroke-dasharray="{dash}"' if dash else ""
        out.append(
            f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{sw}" '
            f'stroke-linecap="round"{da}/>'
        )


def box(x, y, w, h, color, lines, size=20, bold_first=True, dash=None, sw=1.8):
    pts = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    poly = " ".join(f"{px+j(1.2):.1f},{py+j(1.2):.1f}" for px, py in pts)
    out.append(f'<polygon points="{poly}" fill="{FILL[color]}" stroke="none"/>')
    for i in range(4):
        (ax, ay), (bx, by) = pts[i], pts[(i + 1) % 4]
        # excalidraw overshoots its corners a little
        ex, ey = (bx - ax), (by - ay)
        n = math.hypot(ex, ey)
        ox, oy = ex / n * 3, ey / n * 3
        rough_line(ax - ox, ay - oy, bx + ox, by + oy, sw=sw, dash=dash)
    lh = size * 1.25
    y0 = y + h / 2 - (len(lines) - 1) * lh / 2 + size * 0.35
    for i, t in enumerate(lines):
        bold = bold_first and i == 0
        text(x + w / 2, y0 + i * lh, t, size if bold else size - 3, bold=bold)


def runs(t: str) -> str:
    return "".join(
        f'<tspan font-family="DejaVu Sans">{escape(c)}</tspan>' if c in SYMBOLS else escape(c)
        for c in t
    )


def text(x, y, t, size=20, bold=False, anchor="middle", color=INK, rotate=None):
    rot = f' transform="rotate({rotate} {x} {y})"' if rotate else ""
    out.append(
        f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
        f'font-weight="{700 if bold else 400}" text-anchor="{anchor}" fill="{color}"{rot}>'
        f"{runs(t)}</text>"
    )


def arrow(pts, stroke=INK, dash=None, sw=1.8, head=True):
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        rough_line(x1, y1, x2, y2, stroke=stroke, dash=dash, sw=sw)
    if head:
        (x1, y1), (x2, y2) = pts[-2], pts[-1]
        a = math.atan2(y2 - y1, x2 - x1)
        for s in (-1, 1):
            hx = x2 - 14 * math.cos(a + s * 0.45)
            hy = y2 - 14 * math.sin(a + s * 0.45)
            rough_line(hx, hy, x2, y2, stroke=stroke, sw=sw, passes=1)


def frame(x, y, w, h, title, color="#1971c2", dash="10 8"):
    out.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="28" fill="none" '
        f'stroke="{color}" stroke-width="2.6" stroke-dasharray="{dash}"/>'
    )
    text(x + w / 2, y - 12, title, 26, bold=True)


def circle(cx, cy, r, label):
    out.append(
        f'<circle cx="{cx+j()}" cy="{cy+j()}" r="{r}" fill="{FILL["white"]}" '
        f'stroke="{INK}" stroke-width="1.8"/>'
    )
    text(cx, cy + 8, label, 24, bold=True)


def note(x, y, w, h, lines, size=17):
    """Yellow speech bubble, like the hand-written callouts in the reference."""
    out.append(
        f'<path d="M{x+20},{y} H{x+w-20} Q{x+w},{y} {x+w},{y+20} V{y+h-20} '
        f'Q{x+w},{y+h} {x+w-20},{y+h} H{x+70} L{x+30},{y+h+28} L{x+44},{y+h} H{x+20} '
        f'Q{x},{y+h} {x},{y+h-20} V{y+20} Q{x},{y} {x+20},{y} Z" '
        f'fill="{FILL["yellow"]}" stroke="{INK}" stroke-width="1.8"/>'
    )
    for i, t in enumerate(lines):
        text(x + 18, y + 30 + i * size * 1.3, t, size, anchor="start", bold=i == 0)


out.append(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')
text(40, 50, "STREAM ASR v2.5", 34, bold=True, anchor="start")
text(330, 50, "streaming transducer + LM rescoring", 22, anchor="start", color="#495057")

# ---------------------------------------------------------------- acoustic model
frame(20, 100, 1230, 1170, "acoustic model · 63.1 M · one set of weights, offline + streaming")

box(60, 135, 190, 64, "yellow", ["audio", "16 kHz"])
arrow([(258, 167), (318, 167)])
box(325, 135, 220, 64, "yellow", ["log-mel", "80 bins, 10 ms hop"])
arrow([(435, 205), (435, 245), (160, 245), (160, 300)])

frame(40, 280, 1190, 520, "", color="#495057", dash="4 6")
text(1210, 312, "S2 encoder · 61.8 M", 22, bold=True, anchor="end")

# rate lanes
for y, lab in ((392, "25 Hz · d 320"), (472, "12.5 Hz · d 512"), (752, "6.25 Hz · d 640")):
    text(1210 if y < 700 else 769, y, lab, 18, anchor="end" if y < 700 else "middle", color="#1971c2")
for y in (440, 607):
    rough_line(55, y, 1215, y, stroke="#adb5bd", dash="6 8", sw=1.4, passes=1)

box(60, 345, 230, 70, "blue", ["CMVN", "mean fill, pad to 16"])
arrow([(296, 380), (336, 380)])
box(342, 345, 250, 70, "green", ["DwSep subsampling", "×4 → rms_norm"])
arrow([(598, 380), (638, 380)])
box(644, 345, 200, 70, "green", ["S0", "4 blocks"])

arrow([(744, 421), (744, 482)])
text(752, 458, "pool ÷2", 17, anchor="start", color=ORANGE)
box(644, 488, 200, 70, "green", ["S1", "8 blocks"])

arrow([(744, 564), (744, 648)])
text(752, 596, "pool ÷2", 17, anchor="start", color=ORANGE)
box(644, 654, 250, 70, "orange", ["S2", "linear 512 → 640 → 512"])

arrow([(850, 523), (940, 523)])
text(895, 512, "skip", 17, color=ORANGE)
arrow([(900, 689), (965, 689), (965, 548)])
text(972, 640, "upsample ×2", 17, anchor="start", color=ORANGE)
circle(965, 523, 22, "+")
arrow([(989, 523), (1024, 523)])
box(1030, 488, 180, 70, "green", ["S3", "8 blocks"])

arrow([(1120, 564), (1120, 830)])
text(1128, 600, "rms_norm", 17, anchor="start", color=ORANGE)

note(
    58, 448, 572, 140,
    [
        "every block (20 in all)",
        "rms_norm → gated QK-norm RoPE attention → x + β ⊙ rms_norm(·)",
        "rms_norm → dwconv-k15 ⊙ silu MLP → x + β ⊙ rms_norm(·)",
        "β = 0.05 + 0.95 σ(b): the only learnable residual scale",
    ],
    size=17,
)

# ---------------------------------------------------------------- heads
box(990, 836, 240, 70, "blue", ["memory", "12.5 Hz, 80 ms / frame"])
arrow([(1110, 912), (1110, 966)])
box(990, 972, 240, 72, "pink", ["CTC head", "aux loss × 0.5"])
arrow([(984, 871), (690, 975)])

box(60, 980, 260, 72, "purple", ["stateless predictor", "BPE-500 + blank"])
arrow([(326, 1016), (490, 1016)])
box(496, 980, 240, 72, "green", ["additive joiner", "17 cells / frame"])
arrow([(984, 1008), (742, 1008)], stroke=ORANGE, dash="8 7")
text(862, 1040, "sets the band", 18, color=ORANGE)
text(862, 1062, "(cumulative occupancy)", 15, color=ORANGE)

arrow([(560, 1058), (430, 1150)], dash="8 7")
box(250, 1156, 300, 70, "pink", ["banded RNN-T loss", "training: rnnt + 0.5 · ctc"], dash="8 6")
arrow([(680, 1058), (780, 1150)])
box(660, 1156, 320, 70, "green", ["beam search, 10 hyps", "or label-looping greedy"])

text(
    1235, 140, "streaming: 320 ms chunks, 5.12 s left context", 18,
    anchor="end", color=ORANGE,
)
text(1235, 164, "6.9 MB key cache, same output as offline", 18, anchor="end", color=ORANGE)
text(600, 255, "trained on random chunk × left-context masks", 17, anchor="start",
     color=ORANGE, rotate=-2)

# ---------------------------------------------------------------- STREAM-LM
frame(1290, 100, 520, 800, "STREAM-LM · 52.5 M")
box(1390, 140, 320, 64, "yellow", ["hypothesis text", "decoded, then re-encoded"])
arrow([(1550, 210), (1550, 246)])
box(1390, 252, 320, 64, "purple", ["8k unigram tokenizer", "[EOS] + pieces"])
arrow([(1550, 322), (1550, 358)])
box(1390, 364, 320, 60, "blue", ["embedding", "untied"])
arrow([(1550, 430), (1550, 466)])
box(
    1340, 472, 420, 196, "green",
    [
        "16 × causal block, d 512",
        "rms_norm → GQA attention",
        "8 q / 2 kv heads, QK-norm, RoPE",
        "value residual  v + λ · v₀",
        "rms_norm → SwiGLU",
    ],
)
arrow([(1550, 674), (1550, 710)])
box(1390, 716, 320, 64, "blue", ["rms_norm → readout", "zero-init, 8,192 logits"])
text(1550, 830, "trained on LibriSpeech-LM + FineWeb-Edu", 17, color=ORANGE)
text(1550, 852, "+ cv-train, 2.29 B tokens, one pass", 17, color=ORANGE)

# ---------------------------------------------------------------- rescoring
box(
    1290, 960, 520, 100, "orange",
    ["n-best rescoring", "acoustic + α · lm − β · ilm + λ · len"],
    size=22,
)
arrow([(986, 1190), (1240, 1190), (1240, 1030), (1284, 1030)])
text(1110, 1180, "n-best", 18, color=ORANGE)
arrow([(1816, 1000), (1835, 1000), (1835, 172), (1716, 172)], dash="8 7")
text(1852, 590, "each hypothesis", 17, color=ORANGE, rotate=90)
arrow([(1440, 870), (1440, 954)])
text(1448, 925, "lm (log-prob)", 17, anchor="start", color=ORANGE)
text(1540, 1118, "ilm: joiner with memory zeroed", 17, anchor="end", color=ORANGE)
arrow([(1550, 1066), (1550, 1166)])
box(1420, 1172, 260, 64, "green", ["transcript", "A-Z, apostrophe, space"])

svg = (
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
    f'viewBox="0 0 {W} {H}">' + "\n".join(out) + "</svg>"
)
open(sys.argv[1], "w").write(svg)
