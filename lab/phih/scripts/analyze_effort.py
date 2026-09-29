import sys
import numpy as np
from scipy.stats import spearmanr
from common import WORK
from run_effort import load

def least_ndc(efs, ndc, hit, K, target):
    """Least NDC over the ef ladder at which recall@K >= target. Right-censored
    queries (never reach it) are reported separately, never silently capped."""
    ok = hit / K >= target - 1e-9
    reach = ok.any(1)
    first = np.where(reach, ok.argmax(1), len(efs) - 1)
    return np.take_along_axis(ndc, first[:, None], 1).ravel(), reach

if __name__ == '__main__':
  for name in sys.argv[1:]:
      hz = np.load(f'{WORK}/{name}_hardness.npz')
      # queries whose 100 nearest base points are all duplicates of the query itself
      # are degenerate for every distance measure (d_k = 0) -> excluded, and counted
      good = hz['dk'] > 1e-6
      for K in (10, 100):
          efs, ndc, hit = load(f'{WORK}/{name}_effort{K}.bin')
          phi1 = np.fromfile(f'{WORK}/{name}_phi{"" if K==10 else K}.i32',
                             np.int32).reshape(-1, 4)[:, 0]
          print(f'\n[{name}] k={K}  ladder ef {efs.min()}..{efs.max()}  '
                f'mean recall at max ef = {(hit[:, -1]/K).mean():.4f}'
                + (f'  [{(~good).sum()} degenerate queries excluded]' if (~good).any() else ''))
          for t in (0.80, 0.90, 0.98, 1.00):
              L, reach = least_ndc(efs, ndc, hit, K, t)
              if reach.sum() < 50: 
                  print(f'  target {t:.2f}: only {reach.sum()} queries reach it — skipped'); continue
              m = reach & good                            # uncensored, non-degenerate
              row = {'phi_1': phi1, 'LID': hz['lid'], 'RC@k': hz['rck'],
                     'exp': hz['exp2'], 'd_k': hz['dk']}
              rs = {k: spearmanr(v[m], L[m]).statistic for k, v in row.items()}
              print(f'  target {t:.2f}: n={m.sum():5d} reached {reach.mean()*100:5.1f}%  '
                    f'median NDC {np.median(L[m]):7.0f}   ' +
                    '  '.join(f'{k} {rs[k]:+.3f}' for k in ('phi_1','LID','RC@k','exp','d_k')))
