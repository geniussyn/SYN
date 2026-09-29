import os

# corpora root; no default, so a wrong path refuses rather than
# silently reading the wrong file
DATA = os.environ['TC_DATA']
"""audit the ground truth used by run_corpus against the shipped GT files"""
import numpy as np, cfg, run_corpus as rc

# --- GIST: recompute and compare ids with the official ivecs -----------------
b, q, sp = cfg.load_gist()
gi, gd = rc.exact_gt(b, q, sp, kgt=100)
off = np.fromfile(f'{DATA}/gist/gist_groundtruth.ivecs', np.int32)
off = off.reshape(-1, 101)[:, 1:]
same10 = np.mean([len(set(gi[i, :10]) & set(off[i, :10])) for i in range(len(gi))])
print(f'gist: mean |mine ∩ official| @10 = {same10:.3f}/10 over {len(gi)} queries')
d10 = np.array([np.sum((q[i] - b[off[i, 9]])**2) for i in range(len(gi))])
print(f'gist: official d_10 vs mine, max rel gap = '
      f'{np.abs(d10 - gd[:,9]).max()/max(gd[:,9].max(),1e-9):.3e}')
del b, q

# --- YouTube: compare my d_10 with the distances stored in the hdf5 ----------
import h5py
with h5py.File(f'{DATA}/youtube-15M-angular.hdf5','r') as f:
    hd = f['distances'][:]
mine = np.load(f'{cfg.OUT}/youtube_perq.npz')['gt_d10']          # 1 - cos
print(f'youtube: hdf5 d_10 range [{hd[:,9].min():.4f},{hd[:,9].max():.4f}], '
      f'mine [{mine.min():.4f},{mine.max():.4f}]')
for name, conv in (('1-cos', hd[:, 9]), ('cos', 1 - hd[:, 9]),
                   ('sqrt(2-2cos)', np.sqrt(np.maximum(2*hd[:, 9], 0)))):
    print(f'  as {name:13s}: max|Δ| = {np.abs(conv - mine).max():.3e}, '
          f'mean|Δ| = {np.abs(conv - mine).mean():.3e}')
