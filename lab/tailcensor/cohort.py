import numpy as np, cfg, json
for name in ('spacev', 'youtube'):
    z = np.load(f'{cfg.OUT}/{name}_perq.npz'); efs = z['efs']; r = z['recT']; o = z['ordB']
    nq = r.shape[1]; n1, n10 = int(round(nq*.01)), int(round(nq*.10))
    h1, h10, rest = o[:n1], o[:n10], o[n10:]
    print(f'\n### {name}  per-cohort mean recall@10 (cohorts fixed by the independent index)')
    print(f'{"ef":>5} {"all":>7} {"hardest 1%":>11} {"hardest 10%":>12} {"other 90%":>10} {"rec=0 q":>8}')
    for i, ef in enumerate(efs):
        print(f'{ef:5d} {r[i].mean():7.4f} {r[i][h1].mean():11.4f} {r[i][h10].mean():12.4f} '
              f'{r[i][rest].mean():10.4f} {int((r[i]==0).sum()):8d}')
