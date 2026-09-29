# Withdrawn: measurements taken while another lane held the machine

Found by `code/audit_timing.py`, which flags any timed grid where a larger search
budget came back more than 10% faster. That cannot happen on a reserved core; it
is the signature of a second process running during the timed region.

Two causes, both now closed:

1. Until 2026-09-22 07:12 the machine-wide lock covered searches but not builds.
   A 100-thread graph build in one lane therefore ran straight through a pinned,
   timed search in another.
2. Work started outside the queue -- a smoke test, a dependency build -- never
   took the lock at all.

`index_hnsw.timing_lock()` now covers builds as well and prints the load average
on entering a timed region, so the same contention would be visible in the log
rather than only in the data. Everything below was re-measured, not repaired.

| file | worst inversion |
|---|---|
| family_faiss-ivf_msmarco-1m__vanilla__own.json | nprobe 160 -> 192, 25.18 -> 4.96 ms (-80%) |
| family_faiss-ivf_msmarco-1m__syn__own.json     | no inversion, but its window covered an unlocked lane; re-measured as the pair |
| strata_quora__syn__M16__strata_efc200.json     | ef 20 -> 24, 2.04 -> 0.11 ms (-95%) |
| family_vamana_msmarco-1m__syn__own_M8.json     | L 112 -> 128, 0.606 -> 0.412 ms (-32%) |
| family_vamana_msmarco-1m__vanilla__own.json    | L 224 -> 256, 2.335 -> 2.023 ms (-13%) |
| strata_fever__vanilla__M16__strata_efc200.json | ef 56 -> 64, 0.424 -> 0.373 ms (-12%) |
| strata_nq__vanilla__M16__strata_efc200.json    | ef 256 -> 320, 1.426 -> 1.210 ms (-15%) |
