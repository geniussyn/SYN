"""Single-threaded cost of every per-query hardness measure, per corpus.
Charges each measure for the ground truth it actually needs."""
import os, sys, time, subprocess
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[v] = '1'
import numpy as np
from common import _root, CORPORA, WORK
from build import exact_gt
from hardness import dists
from prep_steiner import gt_big, GTSPEC

ST = os.environ.get('PHIH_STEINER_SRC') or (SB + '/src')
SB = os.environ.get('PHIH_STEINER') or _root('PHIH_STEINER', 'the Steiner-hardness checkout (VLDB\'25)')
TMP = os.environ.get('PHIH_TMP', '/tmp/phih')
K = 100


def clock(label, fn):
    t = time.time(); fn(); dt = time.time() - t
    print(f'{label} {dt:.1f}', flush=True)
    return dt


for name in sys.argv[1:]:
    c = CORPORA[name]
    print(f'=== {name} ===', flush=True)
    base, query = c['base'](), c['query']()

    clock(f'{name}.PHI1', lambda: subprocess.run(
        ['./phih', f'{WORK}/{name}', f'{WORK}/{name}_gt{K}.i32', str(K),
         f'{TMP}/t_{name}.i32', '1'], check=True, capture_output=True))

    clock(f'{name}.GT100', lambda: exact_gt(base, query, c['space'], k=K, cand=256, chunk=128))
    clock(f'{name}.DISTPASS', lambda: dists(name))

    gtn, _ = GTSPEC[name]
    clock(f'{name}.GTDEEP{gtn}', lambda: gt_big(base, query, c['space'], gtn,
                                                f'{TMP}/gtdeep_{name}.ivecs'))
    del base, query

    env = dict(os.environ, OMP_NUM_THREADS='1')
    for tag, o in (('DELTA0', '0'), ('ME', '1')):
        clock(f'{name}.{tag}', lambda o=o: subprocess.run(
            [f'{ST}/get_delta0_point', '-d', name, '-b', SB, '-r', '0.98', '-p', '0.98',
             '-k', str(K), '-o', o, '-m', '1'], check=True, capture_output=True, env=env))
print('ALL_TIMING_DONE', flush=True)
