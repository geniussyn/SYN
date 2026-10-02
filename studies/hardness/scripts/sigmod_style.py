"""ACM acmart (SIGMOD) look: Linux Libertine text, Libertine-matched math."""
import matplotlib as mpl

# acmart loads \RequirePackage[tt=false]{libertine}, so Linux
# Libertine is the paper's face and goes first. Libertinus is its
# maintained fork and is kept as a fallback, DejaVu Serif after that
# so a machine with neither still renders rather than substituting
# something arbitrary.
SERIF = 'Linux Libertine O'

def apply():
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': [SERIF, 'Libertinus Serif', 'DejaVu Serif'],
        'mathtext.fontset': 'custom',
        'mathtext.rm': SERIF,
        'mathtext.it': f'{SERIF}:italic',
        'mathtext.bf': f'{SERIF}:bold',
        'mathtext.cal': f'{SERIF}:italic',
        'mathtext.default': 'it',
        'axes.unicode_minus': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42,   # embed real glyphs, not Type 3
    })
