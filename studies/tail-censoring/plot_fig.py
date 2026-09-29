"""black-and-white recall-QPS curves under hard-tail censoring.

modes:  main  3 curves per corpus (all / -1% / -10%), recall vs single-thread QPS
        gain  apparent iso-reported-recall speedup bought by censoring
"""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sigmod_style; sigmod_style.apply()
import cfg

CORP = [('spacev', 'SpaceV100M   100d, $\\ell_2$'),
        ('youtube', 'YouTube15M   1024d, cosine')]
#        key            label                gray   marker  facecolor
SER = [('all',        'all queries',        '0.00', 'o', '0.00'),
       ('drop1_ind',  'hardest 1% removed', '0.42', 's', '0.42'),
       ('drop10_ind', 'hardest 10% removed', '0.62', '^', 'white')]
LW, MS, MEW = 0.9, 4.2, 0.8


def load(name):
    d = json.load(open(f'{cfg.OUT}/{name}_curves.json'))
    out = {}
    for r in d['rows']:
        out.setdefault(r['subset'], []).append((r['ef'], r['recall'], r['qps']))
    return d, {k: np.array(sorted(v)) for k, v in out.items()}


def _frame(ax, title):
    ax.set_title(title, fontsize=10, pad=5)
    ax.grid(True, which='major', lw=0.35, color='0.86')
    ax.grid(True, which='minor', lw=0.25, color='0.94')
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_linewidth(0.6)
    ax.tick_params(width=0.6, labelsize=9)


def curves(ax, out, title):
    for key, lab, gray, mk, mfc in SER:
        a = out[key]
        ax.plot(a[:, 1], a[:, 2], '-', color=gray, lw=LW,
                marker=mk, ms=MS, mfc=mfc, mec=gray, mew=MEW, label=lab)
    ax.set_yscale('log')
    ax.set_xlabel('mean recall@10')
    _frame(ax, title)


def gain(ax, out, title):
    """QPS the all-query set needs to report the censored curve's mean recall"""
    a = out['all']
    r, q = a[:, 1], np.log(a[:, 2])
    for key, lab, gray, mk, mfc in SER[1:]:
        b = out[key]
        ok = (b[:, 1] > r[0]) & (b[:, 1] <= r[-1])
        need = np.exp(np.interp(b[ok, 1], r, q))
        ax.plot(a[ok, 1], a[ok, 2] / need, '-', color=gray, lw=LW, marker=mk,
                ms=MS, mfc=mfc, mec=gray, mew=MEW, label=lab)
    ax.axhline(1.0, color='0.0', lw=0.6, ls=(0, (1, 2)))
    ax.set_xlabel('mean recall@10, all queries')
    _frame(ax, title)


def main(mode='main'):
    names = [(c, t) for c, t in CORP if os.path.exists(f'{cfg.OUT}/{c}_curves.json')]
    fig, axes = plt.subplots(1, len(names), figsize=(3.35 * len(names) + 0.5, 2.85), dpi=260)
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor('white')
    for ax, (name, title) in zip(axes, names):
        d, out = load(name)
        gain(ax, out, title) if mode == 'gain' else curves(ax, out, title)
        ax.legend(frameon=False, fontsize=7.4, loc='lower left' if mode != 'gain' else 'upper left',
                  handlelength=2.4, labelspacing=0.28, borderpad=0.15)
    axes[0].set_ylabel('apparent speedup at equal\nreported recall ($\\times$)'
                       if mode == 'gain' else 'QPS  (single thread)')
    fig.tight_layout(pad=0.4)
    tag = {'main': '', 'gain': '_gain'}[mode]
    for ext in ('pdf', 'png'):
        fig.savefig(f'{cfg.OUT}/tail_censor{tag}.{ext}', bbox_inches='tight', facecolor='white')
    print(f'wrote {cfg.OUT}/tail_censor{tag}.pdf/.png')


if __name__ == '__main__':
    for m in (sys.argv[1:] or ['main', 'gain']):
        main(m)
