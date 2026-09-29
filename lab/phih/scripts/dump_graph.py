"""Read hnswlib's stored index, extract the level-0 adjacency lists,
remap internal ids -> label space, emit forward + reverse CSR."""
import os, sys, struct
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WORK

HDR = '<QQQQQQiIQQQdQ'   # offsetLevel0, max_elem, cur_cnt, size_per_elem, label_off,
                         # offsetData, maxlevel, enterpoint, maxM, maxM0, M, mult, efc

def read_level0(path):
    with open(path, 'rb') as f:
        h = struct.unpack(HDR, f.read(struct.calcsize(HDR)))
        (off0, max_el, n, spe, lab_off, off_data, maxlevel, ep, maxM, maxM0, M, mult, efc) = h
        assert off0 == 0 and struct.calcsize(HDR) == 96, (off0, struct.calcsize(HDR))
        blk = np.frombuffer(f.read(n * spe), dtype=np.uint8).reshape(n, spe)
    meta = dict(n=n, spe=spe, maxM=maxM, maxM0=maxM0, M=M, efc=efc,
                maxlevel=maxlevel, enterpoint=ep, label_off=lab_off, off_data=off_data)
    # link-list header: uint16 count in the low 2 bytes (high 2 bytes = flags)
    cnt = blk[:, 0:2].copy().view(np.uint16).ravel().astype(np.int64)
    flags = blk[:, 2:4].copy().view(np.uint16).ravel()
    nbr = blk[:, 4:4 + 4 * maxM0].copy().view(np.uint32).reshape(n, maxM0)
    label = blk[:, lab_off:lab_off + 8].copy().view(np.uint64).ravel()
    assert (cnt <= maxM0).all(), f'bad degree {cnt.max()} > {maxM0}'
    assert (flags == 0).all(), 'unexpected element flags (deleted?)'
    assert np.array_equal(np.sort(label), np.arange(n, dtype=np.uint64)), 'labels not a permutation'
    return cnt, flags, nbr, label.astype(np.int64), meta

def main(name, M=16, EFC=200):
    path = f'{WORK}/{name}_M{M}_efc{EFC}.bin'
    cnt, flags, nbr, label, meta = read_level0(path)
    n, maxM0 = meta['n'], meta['maxM0']
    print(f'[{name}] {meta}', flush=True)

    valid = np.arange(maxM0)[None, :] < cnt[:, None]
    src_int = np.repeat(np.arange(n, dtype=np.int64), cnt)
    dst_int = nbr[valid].astype(np.int64)
    assert (dst_int < n).all() and (dst_int >= 0).all()

    # internal id -> label
    src = label[src_int]; dst = label[dst_int]
    assert (src != dst).all(), 'self-loop in level-0 graph'
    E = len(src)
    print(f'[{name}] level-0 edges={E}  mean out-deg={E/n:.3f}  '
          f'min={cnt.min()} max={cnt.max()}', flush=True)

    def csr(s, d):
        o = np.argsort(s, kind='stable')
        indices = d[o].astype(np.int32)
        indptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(np.bincount(s, minlength=n), out=indptr[1:])
        return indptr, indices

    fp, fi = csr(src, dst)
    rp, ri = csr(dst, src)
    ind = np.diff(rp)
    print(f'[{name}] mean in-deg={ind.mean():.3f} max={ind.max()} '
          f'zero-in-deg={(ind == 0).sum()}', flush=True)
    for tag, (p, i) in (('fwd', (fp, fi)), ('rev', (rp, ri))):
        p.tofile(f'{WORK}/{name}_{tag}_indptr.i64')
        i.tofile(f'{WORK}/{name}_{tag}_indices.i32')
    np.save(f'{WORK}/{name}_graphmeta.npy', np.array([n, E], dtype=np.int64))

if __name__ == '__main__':
    main(sys.argv[1])
