"""HNSW build + throughput-matched search grid + level-0 graph dump.

Search timing: threads pinned to N_THREADS (default 100) for every arm, so the
x axis of every figure is one comparable quantity. Reported search time per
query is wall clock / n_queries at that thread count; throughput is its inverse.
"""
import argparse, contextlib, json, os, time

# An OpenMP pool that has finished a build does not go away: libgomp keeps its
# team alive and, by default, spins while idle. On the measuring process that is
# a hundred threads burning CPU beside the one pinned thread whose time is being
# reported. Set before anything that links libgomp is imported.
os.environ.setdefault("OMP_WAIT_POLICY", "passive")
os.environ.setdefault("GOMP_SPINCOUNT", "0")
from pathlib import Path
import numpy as np

from paths import EMB, ROOT as LAB                          # noqa: E402
RES = LAB / "results"; WORK = LAB / "work"
# Search is timed on ONE core: a single-threaded latency is the cleanest thing
# to report -- no pool scheduling, no per-query work-stealing jitter, and it is
# the quantity a deployment's tail latency is built out of.
# The reported x axis is the search budget (ef / nprobe), not wall clock, so the
# search no longer has to own the machine: no lock, no reserved core, and as many
# threads as are free. Milliseconds are still recorded, and still useful for a
# rough sense of cost, but they are marked as not a measurement wherever they are
# produced this way. SYN_EXCLUSIVE=1 restores the pinned, locked, single-threaded
# regime for a run whose milliseconds are meant to be quoted.
EXCLUSIVE = os.environ.get("SYN_EXCLUSIVE", "0") == "1"
N_THREADS = int(os.environ.get("SYN_THREADS", "1" if EXCLUSIVE else "32"))
# Build time IS reported now, so every timed build is pinned to the same 100
# threads -- a build on 224 cores and one on 100 are not the same measurement,
# and the arms have to be comparable. Builds that are NOT part of a reported
# number (the hard-decile selector index) use the whole machine instead: they
# are scaffolding, and making them slower buys nothing.
BUILD_THREADS = int(os.environ.get("SYN_BUILD_THREADS", "100"))
BUILD_THREADS_FULL = int(os.environ.get("SYN_BUILD_THREADS_FULL",
                                        str(os.cpu_count() or 100)))
PIN_CORE = int(os.environ.get("SYN_PIN_CORE", (os.cpu_count() or 1) - 1))
# Cores another lane may be using before a timed region starts. One pinned
# search thread is ours; the editor and the OS account for a fraction more.
# share of the PINNED core in use by anything other than this process when a
# timed region is entered. One neighbour thread parked on that core halves
# our throughput, so the bar is low; an idle core reads ~0.00.
LOAD_WARN = float(os.environ.get("SYN_LOAD_WARN", "0.25"))


def core_busy(core, dt=0.2):
    """Busy fraction of ONE core right now, from /proc/stat's per-cpu line.

    The aggregate first line of /proc/stat is the HOST's, not this container's:
    every other tenant on the machine is counted in it, and on a 128-core host
    a few percent of neighbours' work reads as four or five busy cores. That
    made the guard warn on essentially every timed region while the machine
    was, as far as we are concerned, idle. What can actually corrupt a
    measurement is somebody else on the core this search is pinned to, so that
    is what is sampled.

    Called after the lock is taken and before this process starts working, so
    anything it sees belongs to someone else.
    """
    key = f"cpu{core}"

    def snap():
        for line in open("/proc/stat"):
            f = line.split()
            if f and f[0] == key:
                v = [int(x) for x in f[1:9]]
                return sum(v), v[3] + v[4]
        return None, None

    # two samples, take the smaller: the load this lane leaves behind between
    # measurements -- a 4 GB index being unmapped, a page cache being dropped --
    # is a burst of kernel time that decays in well under a second, and
    # reporting it as contention would bury the real thing.
    best = None
    for _ in range(2):
        t0, i0 = snap()
        if t0 is None:
            return 0.0
        time.sleep(dt)
        t1, i1 = snap()
        d = t1 - t0
        v = 0.0 if d <= 0 else 1.0 - (i1 - i0) / d
        best = v if best is None else min(best, v)
    return max(0.0, min(1.0, best))


def cores_busy(dt=0.2):
    """Host-wide busy cores. Kept for reporting; NOT the guard -- see core_busy."""
    def snap():
        v = [int(x) for x in open("/proc/stat").readline().split()[1:9]]
        return sum(v), v[3] + v[4]
    best = None
    for _ in range(2):
        t0, i0 = snap()
        time.sleep(dt)
        t1, i1 = snap()
        d = t1 - t0
        v = 0.0 if d <= 0 else (1.0 - (i1 - i0) / d) * (os.cpu_count() or 1)
        best = v if best is None else min(best, v)
    return best


EF_GRID = [16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224,
           256, 320, 384, 448, 512]


@contextlib.contextmanager
def timing_lock(want=True):
    """Exactly one measured region on this machine at a time.

    Builds take it too. A 100-thread graph build spans every core including the
    reserved one, so a build in one lane and a timed search in another is the
    same contention the lock exists to prevent -- and the queue runs more than
    one lane. Never nest it: two flocks on one file from one process are two open
    file descriptions and the second waits on the first.
    """
    import fcntl
    if not (want and EXCLUSIVE):
        yield
        return
    lock = LAB / "work" / "TIMING_LOCK"
    lock.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock), os.O_CREAT | os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX)
    # The lock only serialises what participates in it. Anything started outside
    # -- a build in a lane that predates this function, a smoke test, a figure --
    # contends silently, and the signature in the data is a larger search budget
    # that came back faster. The load average is the cheapest detector that does
    # not need the other process to cooperate: one pinned search thread should
    # leave the machine near idle.
    b = core_busy(PIN_CORE) if EXCLUSIVE else 0.0
    if b > LOAD_WARN:
        print(f"    !! core {PIN_CORE} is {b:.0%} busy on entering a timed "
              f"region; something outside the lock is on it "
              f"(host-wide: {cores_busy():.1f} cores)", flush=True)
    try:
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN); os.close(fd)


def build(corpus, arm, M, efc, seed=100, index_path=None, threads=None):
    """Index construction. Uses BUILD_THREADS, not the search thread count."""
    import hnswlib
    X = np.load(EMB / f"{corpus}__{arm}_docs.npy", mmap_mode="r")
    n, d = X.shape
    ix = hnswlib.Index(space="ip", dim=d)
    ix.init_index(max_elements=n, ef_construction=efc, M=M, random_seed=seed)
    t0 = time.time()
    chunk = 500_000
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        ix.add_items(np.ascontiguousarray(X[s:e], dtype=np.float32),
                     np.arange(s, e), num_threads=threads or BUILD_THREADS)
        print(f"    added {e:,}/{n:,} ({time.time()-t0:.0f}s)", flush=True)
    bt = time.time() - t0
    if index_path:
        ix.save_index(str(index_path))
    return ix, bt, n, d

BENCH = LAB / "cpp" / "hnsw_bench"


def write_fbin(path, X):
    with open(path, "wb") as f:
        np.array([X.shape[0], X.shape[1]], dtype=np.uint32).tofile(f)
        np.ascontiguousarray(X, dtype=np.float32).tofile(f)


def timed_grid(index_path, Q, ef_grid=EF_GRID, k=10, reps=3, threads=None,
               workdir=None, pin=True, lock=None):
    """Every reported millisecond comes from the native harness in cpp/.

    Timing search through the Python binding would put the query-matrix copy,
    the GIL and the output allocation inside the measurement, which at a few
    hundredths of a millisecond per query is not a rounding error. The binary
    loads the index Python built, runs the grid with a persistent thread pool,
    and times the search loop alone. CPU affinity is inherited from the queue,
    so the thread count and the core set are the same for every arm.
    """
    import json as _json
    import subprocess
    th = threads or N_THREADS
    wd = Path(workdir or Path(index_path).parent)
    qp = wd / (Path(index_path).stem + "_q.fbin")
    pref = wd / (Path(index_path).stem + "_res")
    write_fbin(qp, Q)
    # Pin the search thread: OpenMP is free to migrate a single thread between
    # cores and NUMA nodes mid-run, which on a 38 GB index shows up as timing
    # that is not even monotonic in ef.
    env = {**os.environ, "OMP_PROC_BIND": "true", "OMP_PLACES": "cores",
           "OMP_NUM_THREADS": str(th)}
    # Exactly one timed measurement may run on this machine at a time. The queue
    # is serial, but anything started outside it -- a re-measurement, a smoke
    # test -- would otherwise silently contend and the timing shows it.
    # pin=False is the tuning path: many threads, one pass, milliseconds discarded.
    # It only needs the returned neighbour lists, not the timings.
    #
    # The lock and the pinning are separate concerns and default together only for
    # convenience. The lock buys EXCLUSIVITY -- nothing else on the machine while
    # this runs -- and the pinning buys FIDELITY of the milliseconds. A retrieval
    # that discards its timings still has to hold the lock if a timed cell might
    # be running elsewhere, or it corrupts that cell rather than itself. Pass
    # lock=True, pin=False for exactly that case.
    cmd = [str(BENCH), str(index_path), str(qp), str(k), str(th), str(reps),
           str(pref), ",".join(str(e) for e in ef_grid), "3.0"]
    if not EXCLUSIVE:
        pin = False
    with timing_lock(pin if lock is None else lock):
        # One core is reserved for timing and nothing else runs there. Which
        # core is machine-dependent; the default is the last one.
        pre = ["taskset", "-c", str(PIN_CORE)] if pin and EXCLUSIVE else []
        r = subprocess.run(pre + cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise SystemExit(f"hnsw_bench failed ({r.returncode}): {r.stderr[-2000:]}")
    rows = _json.loads(r.stdout)
    out = []
    for row in rows:
        ef, wall = row["ef"], row["wall_s"]
        lp = Path(str(pref) + f"_ef{ef}.u32")
        lab = np.fromfile(lp, dtype=np.uint32).reshape(len(Q), k).astype(np.int32)
        lp.unlink(missing_ok=True)
        out.append({"ef": ef, "wall_s": wall, "qps": len(Q) / wall,
                    "ms_per_query": wall / len(Q) * 1000.0,
                    "wall_all_reps": row["reps"], "inner": row.get("inner"),
                    "labels": lab})
        print(f"    ef={ef:4d}  {wall/len(Q)*1000:.4f} ms/q  {len(Q)/wall:,.0f} q/s "
              f"[C++ · {th} thread{'s' if th > 1 else ''}"
              f"{'' if pin else ' · TUNING, not a measurement'}]", flush=True)
    qp.unlink(missing_ok=True)
    return out

def measured_out_degree(index_path):
    from hnsw_io import level0_csr_by_label
    ip, idx, h = level0_csr_by_label(index_path)
    return float(np.diff(ip).mean()), ip, idx, h
