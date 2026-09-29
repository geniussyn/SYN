"""ACM acmart (SIGMOD) look: Linux Libertine text, Libertine-matched math."""
import matplotlib as mpl

SERIF = 'Libertinus Serif'

def apply():
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': [SERIF, 'Linux Libertine O', 'DejaVu Serif'],
        'mathtext.fontset': 'custom',
        'mathtext.rm': SERIF,
        'mathtext.it': f'{SERIF}:italic',
        'mathtext.bf': f'{SERIF}:bold',
        'mathtext.cal': f'{SERIF}:italic',
        'mathtext.default': 'it',
        'axes.unicode_minus': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42,   # embed real glyphs, not Type 3
    })
