"""House style for the final figures.

RECONSTRUCTED. This module was never committed, so the seven scripts that import
it could not run. The frame and the four accent colours below are read back from
the pixels of the shipped figures (`figs/fig_recall_distribution.png`,
`fig_hnsw.png`, `fig_efc.png`), which pin them exactly:

    axes background  #EAEAF2   grid white   text #262626
      -> matplotlib's seaborn-v0_8-darkgrid, verbatim
    BLUE   #0000F5   SYN
    RED    #EA3323   Vanilla
    GREEN  #6AA66E   Vanilla, efConstruction sweep (fig_hnsw)
    ORANGE #F2A93C   annotation rules (fig_efc)
    INK    #262626   neutral series

These two lines named the arms the other way round until 2026-09-29. The code
never did -- every figure that draws the two arms has written
`[("vanilla", ..., H.RED), ("syn", ..., H.BLUE)]` from the start -- so the
comment was describing the sampled source figures rather than this palette's
use, and reading it was a way to get the paper's two arms backwards.

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
# Added for the baseline comparison, which needs a fifth accent: the figure
# groups interventions by WHAT they repair, and a reader has to tell four groups
# apart at print size. Purple is the graph-side group (NGFix, RoarGraph); it is
# the one hue left that holds against the other four in greyscale, sitting
# between BLUE and RED in luminance rather than beside either.
PURPLE = "#7B2CBF"

PALETTE = [BLUE, RED, GREEN, ORANGE, INK]

# Marker size and line width, in one place, because they were a magic number
# per figure and a change meant finding all of them. At column width the
# markers do the work of separating series -- a reader picks out a square from
# a triangle long before a solid line from a dashed one -- so they get a step
# up, and the lines come down to stop them smearing into each other where
# eight curves cross.
MARK = 1.28          # multiplies each figure's own per-series marker size
LW = 1.35            # the standard curve; legend swatches stay heavier


def ms(base):
    """A series' marker size, scaled by MARK."""
    return round(base * MARK, 2)


def setup(size=13):
    """seaborn's darkgrid frame at a given base font size."""
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.rcParams.update({
        # ACM camera-ready rejects Type 3 fonts; 42 embeds TrueType instead.
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.dpi": 130, "savefig.dpi": 200, "savefig.bbox": "tight",
        # The venue asks for Helvetica. Helvetica itself is a licensed
        # Monotype face and is not on this machine, so the list names it first
        # and falls through to Nimbus Sans, which is URW's metric-compatible
        # Helvetica and is exactly what LaTeX renders for \usepackage{helvet}.
        # A machine that does have a licensed Helvetica -- macOS ships one --
        # picks it up without any edit here, and the metrics do not move when
        # it does. TeX Gyre Heros is the same clone with wider coverage;
        # Arial and DejaVu are last so a bare machine still renders.
        # font.family is the explicit chain, not the generic alias:
        # matplotlib falls back per glyph only through a list of real
        # family names. The tick labels mark a judged document with a
        # star, which Nimbus Sans has not got -- under the alias that
        # star came out as a last-resort box, and the only warning was
        # on stderr. With the chain, DejaVu supplies that one glyph and
        # every other character still comes from Helvetica or Nimbus.
        "font.family": ["Helvetica", "Helvetica Neue", "Nimbus Sans",
                        "TeX Gyre Heros", "Arial", "DejaVu Sans"],
        "font.sans-serif": ["Helvetica", "Helvetica Neue", "Nimbus Sans",
                            "TeX Gyre Heros", "Arial", "DejaVu Sans"],
        # Maths is set in the same face as the labels rather than in a separate
        # maths family, which is what this module did with Libertine. The
        # generic alias is the point: naming "Nimbus Sans" here would pin the
        # maths to the clone on a machine whose text had resolved to a real
        # Helvetica, and the two would sit side by side in one label.
        # cal/sf/tt are set because a custom fontset leaves them at
        # matplotlib's defaults, and the default for cal is a cursive family
        # nobody here has -- that printed a findfont warning on every figure
        # with maths and silently substituted.
        "mathtext.fontset": "custom",
        "mathtext.rm": "sans",
        "mathtext.it": "sans:italic",
        "mathtext.bf": "sans:bold",
        "mathtext.cal": "sans:italic",
        "mathtext.sf": "sans",
        "mathtext.tt": "monospace",
        "font.size": size,
        "axes.titlesize": size + 2, "axes.labelsize": size,
        "xtick.labelsize": size - 1, "ytick.labelsize": size - 1,
        "legend.fontsize": size - 2,
        "legend.frameon": True, "legend.framealpha": 0.92,
        "legend.edgecolor": "0.8",
        "lines.linewidth": LW, "lines.markersize": ms(5),
        "axes.prop_cycle": plt.cycler(color=PALETTE),
    })
