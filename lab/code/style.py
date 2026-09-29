"""One place for the paper's visual grammar."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, TEAL, VERM = "#0072B2", "#E69F00", "#009E73", "#D55E00"
ENCODER_COLORS = [BLUE, ORANGE, TEAL, VERM]
SYN_C, VAN_C = VERM, "#6E6E6E"          # main figure: vermillion vs grey
M_MARKER = {8: "o", 16: "s", 32: "^", 64: "D"}
M_LS = {8: ":", 16: "-", 32: "--", 64: "-."}

def _also_png(dpi=120):
    """Every figure goes out twice: PDF for the paper, PNG for the live page that
    is watched while the queue runs. Patching savefig once here beats threading a
    second path through four figure scripts and forgetting one of them."""
    from matplotlib.figure import Figure
    if getattr(Figure, "_syn_png_patched", False):
        return
    orig = Figure.savefig

    def savefig(self, fname, *a, **kw):
        r = orig(self, fname, *a, **kw)
        try:
            p = Path(str(fname))
            if p.suffix.lower() == ".pdf":
                d = getattr(Figure, "_syn_png_dpi", dpi)
                orig(self, str(p.with_suffix(".png")), *a, **{**kw, "dpi": d})
        except Exception:
            pass                      # a PNG is a convenience; never fail a figure for it
        return r

    Figure.savefig = savefig
    Figure._syn_png_patched = True


def setup():
    plt.rcParams.update({
        # ACM camera-ready rejects Type 3 fonts; 42 embeds TrueType instead.
        # Matplotlib's PDF default is Type 3, so this has to be set explicitly.
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.dpi": 130, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 19, "axes.titlesize": 21, "axes.labelsize": 20,
        "xtick.labelsize": 17, "ytick.labelsize": 17, "legend.fontsize": 15.5,
        "axes.linewidth": 1.4, "lines.linewidth": 2.8, "lines.markersize": 10,
        "xtick.major.width": 1.4, "ytick.major.width": 1.4,
        "xtick.major.size": 6, "ytick.major.size": 6,
        "axes.grid": True, "grid.alpha": 0.28, "grid.linewidth": 0.9,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": True, "legend.framealpha": 0.93,
        "legend.edgecolor": "0.75",
        "font.family": "DejaVu Sans",
        "font.sans-serif": ["DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
    })


# Importing this module is enough -- fig_digest.py takes only the colours from
# here and sets its own rcParams, and it must emit PNGs too.
_also_png()
