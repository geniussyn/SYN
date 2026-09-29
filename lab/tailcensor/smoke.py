import cfg, numpy as np
cfg.EFS = [16, 32, 64]
cfg.REPS = 2
cfg.OUT = os.environ.get('TC_TMP', '/tmp/tc')
import os; os.makedirs(cfg.OUT, exist_ok=True)
def tiny():
    b = cfg.fvecs(f'{DATA}/gist/gist_base.fvecs')[:40000].copy()
    q = cfg.fvecs(f'{DATA}/gist/gist_query.fvecs')[:300].copy()
    return b, q, 'l2'
def tiny_ip():
    b = cfg._norm_inplace(cfg.fvecs(f'{DATA}/gist/gist_base.fvecs')[:40000].copy())
    q = cfg._norm_inplace(cfg.fvecs(f'{DATA}/gist/gist_query.fvecs')[:300].copy())
    return b, q, 'ip'
cfg.CORPORA = {'tiny': tiny, 'tinyip': tiny_ip}
import run_corpus
run_corpus.EFS = cfg.EFS; run_corpus.REPS = cfg.REPS
run_corpus.KGT = 100
run_corpus.main('tiny')
run_corpus.main('tinyip')
