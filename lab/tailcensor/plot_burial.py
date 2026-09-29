"""the burial picture: the reported average IS the easy 90%."""
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sigmod_style; sigmod_style.apply()
import cfg

CORP = [('spacev', 'SpaceV100M   100d, $\\ell_2$'),
        ('youtube', 'YouTube15M   1024d, cosine')]
LW, MS, MEW = 0.9, 4.2, 0.8

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), dpi=260)
fig.patch.set_facecolor('white')
for ax, (name, title) in zip(axes, CORP):
    z = np.load(f'{cfg.OUT}/{name}_perq.npz')
    efs, R, o = z['efs'], z['recT'], z['ordB']
    nq = R.shape[1]
    n1, n10 = int(round(nq * .01)), int(round(nq * .10))
    series = [('all queries', R.mean(1), '0.00', 'o', '0.00', '-'),
              ('easiest 90%', R[:, o[n10:]].mean(1), '0.00', 'o', 'white', '--'),
              ('hardest 10%', R[:, o[:n10]].mean(1), '0.42', 's', '0.42', '-'),
              ('hardest 1%', R[:, o[:n1]].mean(1), '0.62', '^', 'white', '-')]
    for lab, y, gray, mk, mfc, ls in series:
        ax.plot(efs, y, ls, color=gray, lw=LW, marker=mk, ms=MS, mfc=mfc,
                mec=gray, mew=MEW, label=lab)
    ax.set_xscale('log', base=2)
    ax.set_xticks([16, 64, 256, 1024])
    ax.set_xticklabels(['16', '64', '256', '1024'])
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('ef')
    ax.set_title(title, fontsize=10, pad=5)
    ax.grid(True, which='major', lw=0.35, color='0.86')
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_linewidth(0.6)
    ax.tick_params(width=0.6, labelsize=9)
    ax.legend(frameon=False, fontsize=7.4, loc='lower right', handlelength=2.4,
              labelspacing=0.28, borderpad=0.15)
axes[0].set_ylabel('mean recall@10')
fig.tight_layout(pad=0.4)
for ext in ('pdf', 'png'):
    fig.savefig(f'{cfg.OUT}/tail_burial.{ext}', bbox_inches='tight', facecolor='white')
print(f'wrote {cfg.OUT}/tail_burial.pdf/.png')
