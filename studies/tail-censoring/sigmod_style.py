"""ACM (SIGMOD) look: Helvetica text, Helvetica-matched math."""
import matplotlib as mpl

# The venue asks for Helvetica. Helvetica itself is a licensed Monotype face
# and is absent from most Linux machines, so it goes first and falls through
# to Nimbus Sans -- URW's metric-compatible clone, and exactly what LaTeX
# renders for \usepackage{helvet}. A machine that does have a licensed
# Helvetica, such as macOS, picks it up with no edit here and the metrics do
# not move. TeX Gyre Heros is the same clone with wider coverage; Arial and
# DejaVu are last so a bare machine renders rather than substituting something
# arbitrary.
SANS = ['Helvetica', 'Helvetica Neue', 'Nimbus Sans',
        'TeX Gyre Heros', 'Arial', 'DejaVu Sans']


def apply():
    mpl.rcParams.update({
        # the explicit chain, not the alias: matplotlib falls back
        # per glyph only through a list of real family names
        'font.family': SANS,
        'font.sans-serif': SANS,
        # The generic alias, not a named clone: naming "Nimbus Sans" here would
        # pin the maths to the clone on a machine whose text had resolved to a
        # real Helvetica, and the two would sit side by side in one label.
        'mathtext.fontset': 'custom',
        'mathtext.rm': 'sans',
        'mathtext.it': 'sans:italic',
        'mathtext.bf': 'sans:bold',
        'mathtext.cal': 'sans:italic',
        'mathtext.sf': 'sans',
        'mathtext.tt': 'monospace',
        'mathtext.default': 'it',
        'axes.unicode_minus': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42,   # embed real glyphs, not Type 3
    })
