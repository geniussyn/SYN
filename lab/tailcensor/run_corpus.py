"""
Tail-censoring curves: mean recall@10 vs single-thread QPS, for
  (a) all queries, (b) hardest 1% removed, (c) hardest 10% removed.

Hardness of a query is ranked on an INDEPENDENT index (same params, different
build seed) so that the censored curves are not selecting on their own noise.

Build: 24 threads (lab cap).  Search/timing: 1 thread.
"""
import sys, os, time, json, gc
import cfg
import numpy as np
import hnswlib

K, M, EFC, EFS = cfg.K, cfg.M, cfg.EFC, cfg.EFS
REPS = cfg.REPS
KGT = 100                      # exact neighbours kept (GT@10 + LID tie-break)


def exact_gt(base, query, space, kgt=KGT, step=200_000):
    """blocked exact search; returns (ids, dists) in hnswlib distance convention"""
    n, nq = base.shape[0], query.shape[0]
    bi = np.full((nq, kgt), -1, np.int64)
    bd = np.full((nq, kgt), np.inf, np.float32)
    if space == 'l2':
        qn2 = (query.astype(np.float32) ** 2).sum(1)
    t0 = time.time()
    for i0 in range(0, n, step):
        c = base[i0:i0 + step]
        ip = query @ c.T
        if space == 'l2':
            d = qn2[:, None] - 2.0 * ip + (c ** 2).sum(1)[None, :]
        else:
            d = 1.0 - ip                      # hnswlib 'ip' distance
        m = min(kgt, d.shape[1])
        ci = np.argpartition(d, m - 1, axis=1)[:, :m]
        cd = np.take_along_axis(d, ci, 1)
        bi = np.concatenate([bi, ci.astype(np.int64) + i0], 1)
        bd = np.concatenate([bd, cd.astype(np.float32)], 1)
        o = np.argpartition(bd, kgt - 1, axis=1)[:, :kgt]
        bd = np.take_along_axis(bd, o, 1)
        bi = np.take_along_axis(bi, o, 1)
        del d, ip
    o = np.argsort(bd, axis=1)
    bd = np.take_along_axis(bd, o, 1)
    bi = np.take_along_axis(bi, o, 1)
    print(f'  exact GT k={kgt}: {time.time()-t0:.1f}s', flush=True)
    return bi, bd


def gt_from_ids(base, query, ids, space):
    """distances for a shipped GT id list; also re-sorts each row by distance"""
    nq, kg = ids.shape
    d = np.empty((nq, kg), np.float32)
    t0 = time.time()
    for i in range(nq):
        r = base[ids[i]]
        d[i] = ((query[i][None, :] - r) ** 2).sum(1) if space == 'l2' else 1.0 - r @ query[i]
    o = np.argsort(d, axis=1)
    unsorted = int((o != np.arange(kg)[None, :]).any(1).sum())
    d = np.take_along_axis(d, o, 1)
    ids = np.take_along_axis(ids, o, 1)
    print(f'  shipped GT k={kg}: distances gathered in {time.time()-t0:.1f}s, '
          f'{unsorted}/{nq} rows were not in ascending distance order', flush=True)
    return ids, d


def lid_mle(gd, space, k=100):
    """Amsaleg MLE LID from the k exact neighbour distances (euclidean metric)"""
    r = gd[:, :k].astype(np.float64)
    if space == 'ip':                    # unit norms: ||a-b||^2 = 2-2<a,b> = 2d
        r = np.sqrt(np.maximum(r, 0.0) * 2.0)
    else:
        r = np.sqrt(np.maximum(r, 0.0))
    rk = r[:, -1:].copy()
    rk[rk <= 0] = 1e-12
    w = np.maximum(r[:, :-1], 1e-12) / rk
    s = np.log(w).mean(1)
    out = np.where(s < 0, -1.0 / s, np.inf)
    return out


def recall_rows(ret_d, gt_i, gt_d, ret_i):
    """tie-safe recall@K per query + plain id-intersection recall"""
    thr = gt_d[:, K - 1].astype(np.float64)
    tol = 1e-5 * np.maximum(np.abs(thr), 1e-6) + 1e-7
    safe = (ret_d[:, :K].astype(np.float64) <= (thr + tol)[:, None]).sum(1) / K
    safe = np.minimum(safe, 1.0)
    plain = np.empty(len(gt_i))
    g10 = [set(row) for row in gt_i[:, :K]]
    for i in range(len(gt_i)):
        plain[i] = len(g10[i].intersection(ret_i[i, :K])) / K
    return safe, plain


def build(base, space, seed):
    n, d = base.shape
    ix = hnswlib.Index(space=space, dim=d)
    ix.init_index(max_elements=n, ef_construction=EFC, M=M, random_seed=seed)
    ix.set_num_threads(cfg.NTHREAD)
    t0 = time.time()
    step = 1_000_000
    for i in range(0, n, step):
        ix.add_items(base[i:i + step], np.arange(i, min(n, i + step)))
        print(f'    built {min(n,i+step)}/{n}  {time.time()-t0:.0f}s', flush=True)
    bt = time.time() - t0
    print(f'  build seed={seed}: {bt:.1f}s ({cfg.NTHREAD} thr)', flush=True)
    return ix, bt


def recall_sweep(ix, query, gt_i, gt_d, nthread):
    """per-query recall@10 for every ef (no timing)"""
    ix.set_num_threads(nthread)
    out = {}
    for ef in EFS:
        ix.set_ef(ef)
        li, ld = ix.knn_query(query, k=K, num_threads=nthread)
        safe, plain = recall_rows(ld, gt_i, gt_d, li)
        out[ef] = (safe, plain)
        print(f'    ef={ef:4d} recall={safe.mean():.4f} (plain {plain.mean():.4f})', flush=True)
    return out


def hardness_rank(rec_by_ef, lid):
    mean_rec = np.mean([rec_by_ef[ef][0] for ef in EFS], 0)
    rec_top = rec_by_ef[EFS[-1]][0]
    order = np.lexsort((-lid, rec_top, mean_rec))      # hardest first
    return order, mean_rec


def timed_sweep(ix, query, gt_i, gt_d, subsets):
    ix.set_num_threads(1)
    rows = []
    perq_rec = {}
    for ef in EFS:
        ix.set_ef(ef)
        li, ld = ix.knn_query(query, k=K, num_threads=1)      # warm-up + recall
        safe, plain = recall_rows(ld, gt_i, gt_d, li)
        perq_rec[ef] = safe
        rec = {}
        qs = {nm: np.ascontiguousarray(query[keep]) for nm, keep in subsets.items()}
        secs_all = {nm: [] for nm in subsets}
        for _ in range(REPS):                       # round-robin over subsets
            for nm in subsets:
                t0 = time.perf_counter()
                ix.knn_query(qs[nm], k=K, num_threads=1)
                secs_all[nm].append(time.perf_counter() - t0)
        for name, keep in subsets.items():
            secs = secs_all[name]
            best = min(secs)
            qps = len(keep) / best
            rows.append(dict(ef=ef, subset=name, n=int(len(keep)),
                             recall=float(safe[keep].mean()),
                             recall_plain=float(plain[keep].mean()),
                             qps=float(qps), sec=float(best),
                             sec_reps=[float(x) for x in secs]))
            rec[name] = (safe[keep].mean(), qps)
        msg = '  '.join(f'{k}: r={v[0]:.4f} {v[1]:7.1f}q/s' for k, v in rec.items())
        print(f'    ef={ef:4d}  {msg}', flush=True)
    return rows, perq_rec


def main(name):
    t00 = time.time()
    print(f'=== {name} ===', flush=True)
    loaded = cfg.CORPORA[name]()
    base, query, space = loaded[:3]
    gt_ids = loaded[3] if len(loaded) > 3 else None
    n, d = base.shape
    nq = query.shape[0]
    print(f'  base {base.shape} {base.dtype}  query {query.shape}  space={space}', flush=True)

    if gt_ids is None:
        gt_i, gt_d = exact_gt(base, query, space)
    else:
        gt_i, gt_d = gt_from_ids(base, query, gt_ids, space)
    lid = lid_mle(gt_d, space)

    if os.environ.get('REUSE_B'):
        z0 = np.load(f'{cfg.OUT}/{name}_perq.npz')
        assert list(z0['efs']) == list(EFS), 'saved ef ladder differs'
        recB = {ef: (z0['recB'][i], None) for i, ef in enumerate(EFS)}
        btB = -1.0
        print('  --- index B recall reused from '
              f'{cfg.OUT}/{name}_perq.npz ---', flush=True)
    else:
        print('  --- index B (hardness ranking, independent seed) ---', flush=True)
        ixB, btB = build(base, space, cfg.SEED_B)
        recB = recall_sweep(ixB, query, gt_i, gt_d, cfg.NTHREAD)
        del ixB; gc.collect()

    print('  --- index A (evaluated) ---', flush=True)
    ixA, btA = build(base, space, cfg.SEED_A)
    recA = recall_sweep(ixA, query, gt_i, gt_d, cfg.NTHREAD)

    ordB, meanB = hardness_rank(recB, lid)
    ordA, meanA = hardness_rank(recA, lid)
    n1, n10 = int(round(nq * 0.01)), int(round(nq * 0.10))
    keep_all = np.arange(nq)
    subsets = {
        'all':        keep_all,
        'drop1_ind':  np.sort(ordB[n1:]),
        'drop10_ind': np.sort(ordB[n10:]),
    }
    print(f'  hard sets (ranked on the independent index): '
          f'1% = {n1} q, 10% = {n10} q', flush=True)

    core = os.environ.get('PIN_CORE')
    if core:
        os.sched_setaffinity(0, {int(core)})
        print(f'  timing pinned to core {core}', flush=True)
    print('  --- timed sweep (1 thread) ---', flush=True)
    rows, perq = timed_sweep(ixA, query, gt_i, gt_d, subsets)

    np.savez_compressed(
        f'{cfg.OUT}/{name}_perq.npz',
        lid=lid, meanB=meanB, meanA=meanA, ordB=ordB, ordA=ordA,
        gt_d10=gt_d[:, K - 1],
        recA=np.array([recA[ef][0] for ef in EFS]),
        recB=np.array([recB[ef][0] for ef in EFS]),
        recT=np.array([perq[ef] for ef in EFS]), efs=np.array(EFS))
    json.dump(dict(corpus=name, n=int(n), dim=int(d), nq=int(nq), space=space,
                   M=M, efC=EFC, k=K, reps=REPS, build_sec_A=btA, build_sec_B=btB,
                   seed_A=cfg.SEED_A, seed_B=cfg.SEED_B, threads_build=cfg.NTHREAD,
                   hard_ind_1=[int(x) for x in ordB[:n1]],
                   hard_ind_10=[int(x) for x in ordB[:n10]],
                   rows=rows, total_sec=time.time() - t00),
              open(f'{cfg.OUT}/{name}_curves.json', 'w'), indent=1)
    print(f'  done in {(time.time()-t00)/60:.1f} min -> {cfg.OUT}/{name}_curves.json', flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
