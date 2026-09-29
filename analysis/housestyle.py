"""House style for the final figures.

RECONSTRUCTED. This module was never committed, so the seven scripts that import
it could not run. The frame and the four accent colours below are read back from
the pixels of the shipped figures (`figs/fig_recall_distribution.png`,
`fig_hnsw.png`, `fig_efc.png`), which pin them exactly:

    axes background  #EAEAF2   grid white   text #262626
      -> matplotlib's seaborn-v0_8-darkgrid, verbatim
    BLUE   #0000F5   Vanilla
    RED    #EA3323   SYN
    GREEN  #6AA66E   Vanilla, efConstruction sweep (fig_hnsw)
    ORANGE #F2A93C   annotation rules (fig_efc)
    INK    #262626   neutral series

GREEN and ORANGE are darkened from those readings to #2E7D32 and #D97706. The
originals were chosen against two accents; carrying four families on one axes
needs all four to hold their own at print size, and the sage and the light amber
did not. The other three are unchanged, so every figure that predates the fourth
family looks as it did.

Font sizes are not recoverable from a raster and are set to match the figures at
the sizes the scripts request (`setup(13)`). If the original module turns up,
replace this file with it.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE = "#0000F5"
RED = "#EA3323"
GREEN = "#2E7D32"
ORANGE = "#D97706"
INK = "#262626"

PALETTE = [BLUE, RED, GREEN, ORANGE, INK]


def setup(size=13):
    """seaborn's darkgrid frame at a given base font size."""
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.rcParams.update({
        # ACM camera-ready rejects Type 3 fonts; 42 embeds TrueType instead.
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.dpi": 130, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "font.sans-serif": ["DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": size,
        "axes.titlesize": size + 2, "axes.labelsize": size,
        "xtick.labelsize": size - 1, "ytick.labelsize": size - 1,
        "legend.fontsize": size - 2,
        "legend.frameon": True, "legend.framealpha": 0.92,
        "legend.edgecolor": "0.8",
        "lines.linewidth": 2.0, "lines.markersize": 5,
        "axes.prop_cycle": plt.cycler(color=PALETTE),
    })
