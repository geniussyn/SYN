"""tables + the 'what the censored average buys you' readout"""
import json, sys
import numpy as np
import cfg

def load(name):
    d = json.load(open(f'{cfg.OUT}/{name}_curves.json'))
    cur = {}
    for r in d['rows']:
        cur.setdefault(r['subset'], []).append(r)
    for k in cur:
        cur[k] = sorted(cur[k], key=lambda r: r['ef'])
    return d, cur

def _interp(curve, rec, field):
    r = np.array([c['recall'] for c in curve]); q = np.array([c[field] for c in curve])
    o = np.argsort(r); r, q = r[o], q[o]
    if rec <= r[0] or rec > r[-1]:
        return np.nan
    return float(np.exp(np.interp(rec, r, np.log(q))))

def qps_at_recall(curve, rec):
    """log-linear interpolation of QPS on the all-query curve at a given recall"""
    return _interp(curve, rec, 'qps')

def ef_at_recall(curve, rec):
    return _interp(curve, rec, 'ef')

def report(name):
    d, cur = load(name)
    print(f'\n### {name}  n={d["n"]:,} dim={d["dim"]} nq={d["nq"]} space={d["space"]} '
          f'M={d["M"]} efC={d["efC"]} k={d["k"]}  build {d["build_sec_A"]/60:.1f} min '
          f'({d["threads_build"]} thr)   reps={d["reps"]}, min-of-reps')
    hdr = (f'{"ef":>5} | {"recall all":>10} {"QPS all":>9} | {"rec -1%":>8} {"QPS":>8} '
           f'| {"rec -10%":>8} {"QPS":>8} | {"QPS@equal rec":>14} {"speedup":>8}')
    print(hdr); print('-' * len(hdr))
    for i, r0 in enumerate(cur['all']):
        r1, r10 = cur['drop1_ind'][i], cur['drop10_ind'][i]
        q_eq = qps_at_recall(cur['all'], r10['recall'])
        sp = r0['qps'] / q_eq if q_eq == q_eq else np.nan
        print(f'{r0["ef"]:5d} | {r0["recall"]:10.4f} {r0["qps"]:9.1f} | '
              f'{r1["recall"]:8.4f} {r1["qps"]:8.1f} | {r10["recall"]:8.4f} '
              f'{r10["qps"]:8.1f} | {q_eq:14.1f} {sp:8.2f}x')
    print('  iso-reported-recall price of censoring the tail '
          '(what the all-query set needs to match the censored mean recall):')
    for i, r0 in enumerate(cur['all']):
        for key, tag in (('drop1_ind', ' -1%'), ('drop10_ind', '-10%')):
            rc = cur[key][i]
            qe, ee = qps_at_recall(cur['all'], rc['recall']), ef_at_recall(cur['all'], rc['recall'])
            if qe == qe:
                print(f'    ef={r0["ef"]:4d} {tag}: recall {r0["recall"]:.4f}->{rc["recall"]:.4f}'
                      f'  all-query needs ef~{ee:6.0f} ({ee/r0["ef"]:4.1f}x) -> '
                      f'{qe:7.1f} q/s vs {r0["qps"]:7.1f} = {r0["qps"]/qe:4.2f}x apparent speedup')
            else:
                print(f'    ef={r0["ef"]:4d} {tag}: recall {r0["recall"]:.4f}->{rc["recall"]:.4f}'
                      f'  beyond the all-query ladder (ef<={cur["all"][-1]["ef"]})')
    # timing dispersion
    disp = [max(r['sec_reps']) / min(r['sec_reps']) - 1 for r in d['rows']]
    print(f'  timing spread (max/min-1) over all {len(disp)} timed batches: '
          f'median {np.median(disp)*100:.1f}%  p90 {np.percentile(disp,90)*100:.1f}%')
    z = np.load(f'{cfg.OUT}/{name}_perq.npz')
    n1, n10 = int(round(d['nq']*0.01)), int(round(d['nq']*0.10))
    ordB = z['ordB']
    recT = z['recT']; efs = z['efs']
    for ef_t in (128, efs[-1]):
        j = int(np.where(efs == ef_t)[0][0])
        r = recT[j]
        print(f'  ef={ef_t}: all {r.mean():.4f} | hardest 1% {r[ordB[:n1]].mean():.4f} '
              f'| hardest 10% {r[ordB[:n10]].mean():.4f} | zero-recall queries '
              f'{int((r==0).sum())}')

if __name__ == '__main__':
    for n in (sys.argv[1:] or ['spacev', 'youtube']):
        try:
            report(n)
        except FileNotFoundError as e:
            print(f'{n}: not available yet ({e})')
