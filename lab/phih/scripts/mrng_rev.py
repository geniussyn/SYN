"""MRNG file: per node [uint32 deg][deg x uint32 ids].
_reversed_std: per node, uint32 ids terminated by 0xFFFFFFFF."""
import sys
import numpy as np

def read_mrng(path):
    a = np.fromfile(path, dtype=np.uint32)
    src, dst, i, n = [], [], 0, 0
    while i < len(a):
        k = int(a[i]); i += 1
        if k:
            dst.append(a[i:i+k]); src.append(np.full(k, n, dtype=np.uint32))
        i += k; n += 1
    return n, (np.concatenate(src) if src else np.empty(0, np.uint32)), \
              (np.concatenate(dst) if dst else np.empty(0, np.uint32))

def write_rev(n, src, dst, out):
    o = np.argsort(dst, kind='stable')          # group by destination -> reverse adjacency
    rid = src[o]
    deg = np.bincount(dst.astype(np.int64), minlength=n)
    buf = np.empty(len(rid) + n, dtype=np.uint32)
    pos = np.cumsum(deg + 1) - 1
    m = np.ones(len(buf), dtype=bool); m[pos] = False
    buf[m] = rid; buf[pos] = np.uint32(0xFFFFFFFF)
    buf.tofile(out)
    return deg

if __name__ == '__main__':
    p = sys.argv[1]
    n, src, dst = read_mrng(p)
    deg = write_rev(n, src, dst, p + '_reversed_std')
    od = np.bincount(src.astype(np.int64), minlength=n)
    print(f'{p}: n={n} edges={len(src)} out-deg mean={od.mean():.2f} max={od.max()} '
          f'in-deg mean={deg.mean():.2f} max={deg.max()} zero-in={(deg==0).sum()}')
