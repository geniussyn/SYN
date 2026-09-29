// Native timing for NSG search. The index is built and written by faiss's Python
// side (construction is not the timed quantity); this loads it and times the
// search loop with faiss's own C++ path, so no binding overhead lands in a
// measurement whose budget is hundredths of a millisecond per query.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <omp.h>
#include <faiss/IndexNSG.h>
#include <faiss/index_io.h>

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
                        "<out_prefix> <L1,L2,...> [min_sec]\n", argv[0]);
        return 1;
    }
    const size_t k = atoi(argv[3]);
    const int threads = atoi(argv[4]), reps = atoi(argv[5]);
    const std::string out_prefix = argv[6];
    // Repeat the sweep until the timed region is long enough that OpenMP
    // region entry and scheduler jitter cannot move it. Without this a
    // cheap ef on a small query set measures the runtime, not the search.
    const double min_sec = argc > 8 ? atof(argv[8]) : 1.0;
    std::vector<int> Ls;
    for (char *t = strtok(argv[7], ","); t; t = strtok(nullptr, ",")) Ls.push_back(atoi(t));

    size_t nq, dim;
    std::vector<float> Q = read_fbin(argv[2], nq, dim);
    faiss::Index *base = faiss::read_index(argv[1]);
    auto *idx = dynamic_cast<faiss::IndexNSG *>(base);
    if (!idx) { fprintf(stderr, "not an IndexNSG\n"); return 3; }
    omp_set_num_threads(threads);
    fprintf(stderr, "loaded %s: ntotal %zd, dim %zu, %zu queries\n",
            argv[1], (ssize_t)idx->ntotal, dim, nq);

    std::vector<faiss::idx_t> I(nq * k);
    std::vector<float> D(nq * k);
    idx->nsg.search_L = 64;
    idx->search(nq, Q.data(), k, D.data(), I.data());     // warm the page cache

    printf("[\n");
    for (size_t e = 0; e < Ls.size(); ++e) {
        idx->nsg.search_L = Ls[e];
        auto c0 = std::chrono::steady_clock::now();
        idx->search(nq, Q.data(), k, D.data(), I.data());  // calibration pass
        double t1 = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - c0).count();
        int inner = (int)std::max(1.0, std::ceil(min_sec / std::max(t1, 1e-9)));
        std::vector<double> times;
        for (int r = 0; r < reps; ++r) {
            auto t0 = std::chrono::steady_clock::now();
            for (int q = 0; q < inner; ++q)
                idx->search(nq, Q.data(), k, D.data(), I.data());
            times.push_back(std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count() / inner);
        }
        std::sort(times.begin(), times.end());
        double med = times[0];          // fastest pass: see hnsw_bench.cpp
        char p[4096];
        snprintf(p, sizeof p, "%s_L%d.u32", out_prefix.c_str(), Ls[e]);
        FILE *f = fopen(p, "wb");
        std::vector<uint32_t> out(nq * k);
        for (size_t i = 0; i < nq * k; ++i) out[i] = I[i] < 0 ? 0u : (uint32_t)I[i];
        fwrite(out.data(), 4, nq * k, f);
        fclose(f);
        printf("  {\"L\": %d, \"wall_s\": %.9f, \"n_queries\": %zu, \"inner\": %d, \"reps\": [",
               Ls[e], med, nq, inner);
        for (size_t i = 0; i < times.size(); ++i)
            printf("%.9f%s", times[i], i + 1 < times.size() ? ", " : "");
        printf("]}%s\n", e + 1 < Ls.size() ? "," : "");
        fflush(stdout);
    }
    printf("]\n");
    delete base;
    return 0;
}
