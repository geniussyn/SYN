"""Parse hnswlib's saved index: level-0 adjacency + labels.

hnswlib saveIndex() writes, in order (writeBinaryPOD):
  offsetLevel0_(size_t) max_elements_(size_t) cur_element_count(size_t)
  size_data_per_element_(size_t) label_offset_(size_t) offsetData_(size_t)
  maxlevel_(int) enterpoint_node_(uint32) maxM_(size_t) maxM0_(size_t)
  M_(size_t) mult_(double) ef_construction_(size_t)
then data_level0_memory_ (cur_element_count * size_data_per_element_ bytes).
Each level-0 record: [uint32 n_links][maxM0_ x uint32 link][vector data][uint64 label].
"""
import struct
import numpy as np

HDR = struct.Struct("<6q i I 3q d q")   # 6*8 + 4 + 4 + 24 + 8 + 8 = 96 bytes

def read_header(path):
    with open(path, "rb") as f:
        b = f.read(HDR.size)
    (off0, maxel, cur, sdpe, laboff, datoff, maxlevel, ep, maxM, maxM0, M, mult, efc) = HDR.unpack(b)
    return dict(offsetLevel0=off0, max_elements=maxel, cur_element_count=cur,
                size_data_per_element=sdpe, label_offset=laboff, offsetData=datoff,
                maxlevel=maxlevel, enterpoint=ep, maxM=maxM, maxM0=maxM0, M=M,
                mult=mult, ef_construction=efc, header_bytes=HDR.size)

def level0_csr(path):
    """Returns (indptr, indices, labels, header). Node ids are hnswlib INTERNAL ids."""
    h = read_header(path)
    n, sdpe, maxM0 = h["cur_element_count"], h["size_data_per_element"], h["maxM0"]
    raw = np.memmap(path, dtype=np.uint8, mode="r",
                    offset=h["header_bytes"], shape=(n, sdpe))
    link_block = raw[:, : (maxM0 + 1) * 4].reshape(n, -1)
    lb = np.ascontiguousarray(link_block).view(np.uint32).reshape(n, maxM0 + 1)
    deg = lb[:, 0].astype(np.int64)
    if deg.max() > maxM0:
        raise ValueError(f"degree {deg.max()} > maxM0 {maxM0}: header layout mismatch")
    indptr = np.zeros(n + 1, dtype=np.int64); np.cumsum(deg, out=indptr[1:])
    indices = np.empty(int(indptr[-1]), dtype=np.int32)
    cols = np.arange(maxM0, dtype=np.int64)
    keep = cols[None, :] < deg[:, None]
    indices[:] = lb[:, 1:][keep].astype(np.int32)
    labs = np.ascontiguousarray(raw[:, h["label_offset"]: h["label_offset"] + 8]).view(np.uint64).ravel()
    return indptr, indices, labs.astype(np.int64), h

def level0_csr_by_label(path):
    """Same, remapped so node id == the label passed to add_items.

    hnswlib assigns internal ids under a lock during parallel add_items, so the
    internal order is NOT the insertion order; every downstream use (GT sets,
    qrels, exact top-k) is in label space, so we remap once here.
    """
    indptr, indices, labels, h = level0_csr(path)
    n = len(labels)
    if np.array_equal(labels, np.arange(n)):
        return indptr, indices, h
    perm = np.empty(n, dtype=np.int64); perm[labels] = np.arange(n)   # internal -> label
    order = np.argsort(labels, kind="stable")                        # label -> internal
    nd = np.diff(indptr)[order]
    nip = np.zeros(n + 1, dtype=np.int64); np.cumsum(nd, out=nip[1:])
    total = int(nip[-1])
    gather = np.repeat(indptr[order] - nip[:-1], nd) + np.arange(total, dtype=np.int64)
    ni = perm[indices[gather]].astype(np.int32)
    return nip, ni, h
