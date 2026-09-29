"""Replacement/extension for the paper's Figure 3 (§3.3), drawn at final acmart
\textwidth. Adds Steiner ME as a fifth measure and an effort-axis panel.
Only k=100 is shown: phi_h <= k, so at k=10 the index has ten levels and the
comparison is resolution-limited (the paper already says this)."""
import glob, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sigmod_style; sigmod_style.apply()
from matplotlib.patches import Patch
from scipy.stats import spearmanr, rankdata, pearsonr
from common import _root, WORK
from run_effort import load
from analyze_effort import least_ndc

matplotlib.rcParams['hatch.linewidth'] = 0.45
import os
SB = os.environ.get('PHIH_STEINER') or _root('PHIH_STEINER', 'the Steiner-hardness checkout (VLDB\'25)')
TEXTWIDTH, K, EF, ACC = 7.0, 100, 128, 0.98
# The middle corpus is the ann-benchmarks MS MARCO v1 embedded with OpenAI
# text-embedding-ada-002. It is NOT the paper's MS MARCO 1M, which is a BEIR
# subset under BGE-M3, so it carries the encoder in its label to keep the two
# apart on the page.
CORP = [('glove', 'GloVe'), ('msmarco', 'MS MARCO\nada-002'),
        ('youtube1m', 'YouTube-1M')]
MEAS = [('phi', '$\\varphi_1$',     '#FFFFFF', '///'),
        ('me',  'Steiner ME',       '#FFFFFF', 'ooo'),
        ('lid', 'LID',              '#DCDCDC', '\\\\\\'),
        ('exp', '$d_{2k}/d_k$',     '#DCDCDC', 'xxx'),
        ('rck', 'RC@$k$',           '#9A9A9A', '')]
INK, INK2 = 'black', '#4D4D4D'

def partial(a, b, ctrl):
    A, B = rankdata(a).astype(float), rankdata(b).astype(float)
    C = np.column_stack([np.ones(len(A)), rankdata(ctrl).astype(float)])
    r = lambda v: v - C @ np.linalg.lstsq(C, v, rcond=None)[0]
    return pearsonr(r(A), r(B))[0]

DATA = {}
for name, _ in CORP:
    hz = np.load(f'{WORK}/{name}_hardness.npz')
    me = np.fromfile(glob.glob(
        f'{SB}/{name}/{name}_me_exhausted*recall0.98_prob0.98_k{K}_*hnsw*')[0],
        np.int32).astype(float)
    phi = np.fromfile(f'{WORK}/{name}_phi{K}.i32', np.int32).reshape(-1, 4)[:, 0]
    efs, ndc, hit = load(f'{WORK}/{name}_effort{K}.bin')
    keep = (hz['dk'] > 1e-6) & (me > 0)
    rec = hit[:, int(np.where(efs == EF)[0][0])] / K
    L, reach = least_ndc(efs, ndc, hit, K, ACC)
    V = {'phi': phi, 'me': me, 'lid': hz['lid'], 'exp': hz['exp2'], 'rck': hz['rck']}
    DATA[name] = dict(V=V, keep=keep, rec=rec, L=L, m=keep & reach, n=int(keep.sum()),
                      meanrec=float(rec[keep].mean()), medL=float(np.median(L[keep & reach])))

def panel(ax, vals, ylab, ncorp=3, w=0.165):
    x = np.arange(ncorp)
    for i, (k, lab, face, hat) in enumerate(MEAS):
        ax.bar(x + (i - 2) * w, vals[k], w * 0.88, facecolor=face, hatch=hat,
               edgecolor='black', linewidth=0.5, zorder=3)
        for xi, v in zip(x + (i - 2) * w, vals[k]):
            ax.text(xi, v + 0.015, f'{v:.2f}'.lstrip('0'), ha='center', va='bottom',
                    fontsize=5.4, color=INK, zorder=4)
    ax.set_ylim(0, 1.16); ax.set_yticks([0, 0.5, 1.0]); ax.set_yticklabels(['0', '.5', '1'])
    ax.set_xlim(-0.55, ncorp - 0.45)
    ax.set_xticks(x); ax.set_xticklabels([c[1] for c in CORP], fontsize=7)
    ax.set_ylabel(ylab, fontsize=7.5, labelpad=2)
    ax.grid(axis='y', color='#DDDDDD', lw=0.4, zorder=0); ax.set_axisbelow(True)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'): ax.spines[s].set_color('#999999'); ax.spines[s].set_linewidth(0.5)
    ax.tick_params(direction='out', length=2, width=0.5, colors=INK2, labelsize=7, pad=1.5)
    for t in ax.get_xticklabels(): t.set_color(INK)

fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.15), dpi=600)
fig.patch.set_facecolor('white')

A = {k: [abs(spearmanr(DATA[n]['V'][k][DATA[n]['keep']],
                       DATA[n]['rec'][DATA[n]['keep']]).statistic) for n, _ in CORP]
     for k, *_ in MEAS}
B = {k: [abs(spearmanr(DATA[n]['V'][k][DATA[n]['m']],
                       DATA[n]['L'][DATA[n]['m']]).statistic) for n, _ in CORP]
     for k, *_ in MEAS}
panel(axes[0], A, f'$|\\rho|$ with recall@{K}')
panel(axes[1], B, '$|\\rho|$ with least NDC')
axes[0].set_title(f'(a) fixed budget (ef={EF})', fontsize=8, pad=5)
axes[1].set_title(f'(b) effort to recall {ACC}', fontsize=8, pad=5)

# (c) partial correlations: does phi_1 add over LID, and over Steiner ME?
ax = axes[2]
PAIRS = [('$\\varphi_1|$LID', '#FFFFFF', '...'), ('LID$|\\varphi_1$', '#BFBFBF', '...'),
         ('$\\varphi_1|$ME',  '#FFFFFF', '++'),  ('ME$|\\varphi_1$',  '#BFBFBF', '++')]
P = []
for n, _ in CORP:
    d = DATA[n]; kp = d['keep']
    P.append([partial(d['V']['phi'][kp], d['rec'][kp], d['V']['lid'][kp]),
              partial(d['V']['lid'][kp], d['rec'][kp], d['V']['phi'][kp]),
              partial(d['V']['phi'][kp], d['rec'][kp], d['V']['me'][kp]),
              partial(d['V']['me'][kp],  d['rec'][kp], d['V']['phi'][kp])])
P = np.array(P)
x, w = np.arange(3), 0.2
for i, (lab, face, hat) in enumerate(PAIRS):
    ax.bar(x + (i - 1.5) * w, np.abs(P[:, i]), w * 0.88, facecolor=face, hatch=hat,
           edgecolor='black', linewidth=0.5, zorder=3)
    for xi, v in zip(x + (i - 1.5) * w, P[:, i]):
        ax.text(xi, abs(v) + 0.015, f'{v:+.2f}'.replace('0.', '.'), ha='center',
                va='bottom', fontsize=5.4, color=INK, zorder=4)
ax.set_ylim(0, 1.40); ax.set_yticks([0, 0.5, 1.0]); ax.set_yticklabels(['0', '.5', '1'])
ax.set_xlim(-0.55, 2.45); ax.set_xticks(x)
ax.set_xticklabels([c[1] for c in CORP], fontsize=7)
ax.set_ylabel('partial $|\\rho|$', fontsize=7.5, labelpad=2)
ax.set_title('(c) partial, fixed budget', fontsize=8, pad=5)
ax.grid(axis='y', color='#DDDDDD', lw=0.4, zorder=0); ax.set_axisbelow(True)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
for s in ('left', 'bottom'): ax.spines[s].set_color('#999999'); ax.spines[s].set_linewidth(0.5)
ax.tick_params(direction='out', length=2, width=0.5, colors=INK2, labelsize=7, pad=1.5)
for t in ax.get_xticklabels(): t.set_color(INK)

ax.legend(handles=[Patch(facecolor=f, hatch=h, edgecolor='black', linewidth=0.5, label=l)
                   for l, f, h in PAIRS],
          loc='upper center', ncol=2, frameon=False, fontsize=6.0,
          handlelength=1.2, columnspacing=0.8, handletextpad=0.3,
          labelspacing=0.2, borderpad=0.1)
fig.legend(handles=[Patch(facecolor=f, hatch=h, edgecolor='black', linewidth=0.5, label=l)
                    for _, l, f, h in MEAS],
           loc='lower center', ncol=5, frameon=False, fontsize=7,
           bbox_to_anchor=(0.34, -0.04), handlelength=1.3, columnspacing=1.1,
           handletextpad=0.3)
fig.text(0.5, -0.055,
 "ada-002 = OpenAI text-embedding-ada-002 on ann-benchmarks MS MARCO v1; "
 "not the BEIR subset under BGE-M3 used elsewhere in this paper.",
 ha="center", fontsize=5.8, color="#4D4D4D")
fig.tight_layout(rect=(0, 0.075, 1, 1.0))
for e in ('png', 'pdf'):
    fig.savefig(f'./fig3_hardness_measures.{e}', facecolor='white',
                bbox_inches='tight', pad_inches=0.01)
print('wrote paper_fig3  %.2f x %.2f in' % tuple(fig.get_size_inches()))

# ---- values ----
print('\n### per-corpus context')
for n, disp in CORP:
    d = DATA[n]
    print(f'  {n:10s} n={d["n"]:5d}  mean recall@{K}@ef{EF}={d["meanrec"]:.4f}  '
          f'median least NDC@{ACC}={d["medL"]:,.0f}')
print('\n### (a) |rho| with recall@%d at ef=%d' % (K, EF))
for k, lab, *_ in MEAS: print(f'  {lab:14s} ' + '  '.join(f'{v:.3f}' for v in A[k]))
print('\n### (b) |rho| with least NDC to %.2f' % ACC)
for k, lab, *_ in MEAS: print(f'  {lab:14s} ' + '  '.join(f'{v:.3f}' for v in B[k]))
print('\n### (c) partial |rho|, fixed budget')
for i, (lab, *_) in enumerate(PAIRS): print(f'  {lab:14s} ' + '  '.join(f'{v:+.3f}' for v in P[:, i]))
print('\n### phi_1 <-> Steiner ME agreement (Spearman)')
for n, _ in CORP:
    d = DATA[n]; kp = d['keep']
    print(f'  {n:10s} {spearmanr(d["V"]["phi"][kp], d["V"]["me"][kp]).statistic:+.3f}')
