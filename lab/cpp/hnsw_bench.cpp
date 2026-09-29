// Native timing for HNSW search. The Python binding copies the query matrix,
// takes the GIL around the result marshalling and allocates the output arrays;
// at ~0.03 ms/query that is not a rounding error, so every reported millisecond
// comes from here instead.
//
// Build only measures nothing here: the index is built and saved by Python, and
// this process loads it, runs the ef grid, and reports wall time around the
// search loop alone.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <omp.h>
#include <vector>
#include "hnswlib/hnswlib.h"

static std::vector<float> read_fbin(const char *path, size_t &n, size_t &d) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    uint32_t a, b;
    if (fread(&a, 4, 1, f) != 1 || fread(&b, 4, 1, f) != 1) exit(2);
    n = a; d = b;
    std::vector<float> v(n * d);
    if (fread(v.data(), 4, n * d, f) != n * d) exit(2);
    fclose(f);
    return v;
}

int main(int argc, char **argv) {
    if (argc < 8) {
        fprintf(stderr, "usage: %s <index> <query.fbin> <k> <threads> <reps> "
                        "<out_prefix> <ef1,ef2,...> [min_sec]\n", argv[0]);
        return 1;
    }
    const char *idx_path = argv[1], *q_path = argv[2];
    const size_t k = atoi(argv[3]);
    const int threads = atoi(argv[4]), reps = atoi(argv[5]);
    const std::string out_prefix = argv[6];
    // Repeat the sweep until the timed region is long enough that OpenMP
    // region entry and scheduler jitter cannot move it. Without this a
    // cheap ef on a small query set measures the runtime, not the search.
    const double min_sec = argc > 8 ? atof(argv[8]) : 1.0;
    std::vector<size_t> efs;
    for (char *t = strtok(argv[7], ","); t; t = strtok(nullptr, ",")) efs.push_back(atoi(t));

    size_t nq, dim;
    std::vector<float> Q = read_fbin(q_path, nq, dim);
    hnswlib::InnerProductSpace space(dim);
    hnswlib::HierarchicalNSW<float> idx(&space, idx_path, false, 0, false);
    fprintf(stderr, "loaded %s: %zu elements, dim %zu, %zu queries\n",
            idx_path, idx.cur_element_count.load(), dim, nq);

    std::vector<hnswlib::labeltype> lab(nq * k);

    // OpenMP keeps its thread team alive between parallel regions, so neither
    // thread creation nor a hand-rolled spin barrier lands inside a measurement
    // whose whole budget is hundredths of a millisecond per query.
    omp_set_num_threads(threads);
    omp_set_dynamic(0);

    auto sweep = [&](size_t ef) {
        idx.setEf(ef);
#pragma omp parallel for schedule(dynamic, 16)
        for (long long i = 0; i < (long long)nq; ++i) {
            auto r = idx.searchKnn(Q.data() + (size_t)i * dim, k);
            // searchKnn returns a max-heap: farthest first, so fill backwards
            size_t j = k;
            while (!r.empty() && j > 0) { lab[(size_t)i * k + (--j)] = r.top().second; r.pop(); }
            while (j > 0) lab[(size_t)i * k + (--j)] = 0;   // fewer than k found
        }
    };

    sweep(64);                                              // warm the page cache

    printf("[\n");
    for (size_t e = 0; e < efs.size(); ++e) {
        auto c0 = std::chrono::steady_clock::now();
        sweep(efs[e]);                                    // calibration pass
        double t1 = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - c0).count();
        int inner = (int)std::max(1.0, std::ceil(min_sec / std::max(t1, 1e-9)));
        std::vector<double> times;
        for (int r = 0; r < reps; ++r) {
            auto t0 = std::chrono::steady_clock::now();
            for (int q = 0; q < inner; ++q) sweep(efs[e]);
            times.push_back(std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count() / inner);
        }
        std::sort(times.begin(), times.end());
        // MINIMUM, not median: a single-threaded search over a 38 GB index is
        // exposed to any transient on the machine, and every transient can only
        // make a pass slower. The fastest pass is the least contaminated
        // estimate of what the index actually costs.
        double med = times[0];
        char p[4096];
        snprintf(p, sizeof p, "%s_ef%zu.u32", out_prefix.c_str(), efs[e]);
        FILE *f = fopen(p, "wb");
        std::vector<uint32_t> out(nq * k);
        for (size_t i = 0; i < nq * k; ++i) out[i] = (uint32_t)lab[i];
        fwrite(out.data(), 4, nq * k, f);
        fclose(f);
        printf("  {\"ef\": %zu, \"wall_s\": %.9f, \"n_queries\": %zu, \"inner\": %d, \"reps\": [",
               efs[e], med, nq, inner);
        for (size_t i = 0; i < times.size(); ++i)
            printf("%.9f%s", times[i], i + 1 < times.size() ? ", " : "");
        printf("]}%s\n", e + 1 < efs.size() ? "," : "");
        fflush(stdout);
    }
    printf("]\n");
    return 0;
}
