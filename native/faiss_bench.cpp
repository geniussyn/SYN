// Single-threaded search timing for any faiss index (HNSW, NSG, IVF).
// Construction happens in Python; this loads the written index and times the
// search loop alone, so no binding overhead lands in a reported millisecond.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <omp.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexNSG.h>
#include <faiss/index_io.h>

static std::vector<float> read_fbin(const char *p, size_t &n, size_t &d) {
    FILE *f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", p); exit(2); }
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
                        "<out_prefix> <p1,p2,...> [min_sec]\n", argv[0]);
        return 1;
    }
    const size_t k = atoi(argv[3]);
    const int threads = atoi(argv[4]), reps = atoi(argv[5]);
    const std::string pref = argv[6];
    const double min_sec = argc > 8 ? atof(argv[8]) : 3.0;
    std::vector<int> P;
    for (char *t = strtok(argv[7], ","); t; t = strtok(nullptr, ",")) P.push_back(atoi(t));

    size_t nq, dim;
    std::vector<float> Q = read_fbin(argv[2], nq, dim);
    faiss::Index *ix = faiss::read_index(argv[1]);
    omp_set_num_threads(threads);

    auto *hnsw = dynamic_cast<faiss::IndexHNSW *>(ix);
    auto *nsg  = dynamic_cast<faiss::IndexNSG *>(ix);
    auto *ivf  = dynamic_cast<faiss::IndexIVF *>(ix);
    const char *kind = hnsw ? "HNSW" : nsg ? "NSG" : ivf ? "IVF" : "?";
    if (!hnsw && !nsg && !ivf) { fprintf(stderr, "unsupported index type\n"); return 3; }
    fprintf(stderr, "loaded %s (%s): ntotal %zd, dim %zu, %zu queries\n",
            argv[1], kind, (ssize_t)ix->ntotal, dim, nq);

    auto set_param = [&](int v) {
        if (hnsw) hnsw->hnsw.efSearch = v;
        else if (nsg) nsg->nsg.search_L = v;
        else ivf->nprobe = v;
    };
    std::vector<faiss::idx_t> I(nq * k);
    std::vector<float> D(nq * k);
    set_param(P.empty() ? 64 : P[0]);
    ix->search(nq, Q.data(), k, D.data(), I.data());          // warm

    printf("[\n");
    for (size_t e = 0; e < P.size(); ++e) {
        set_param(P[e]);
        auto c0 = std::chrono::steady_clock::now();
        ix->search(nq, Q.data(), k, D.data(), I.data());
        double t1 = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - c0).count();
        int inner = (int)std::max(1.0, std::ceil(min_sec / std::max(t1, 1e-9)));
        std::vector<double> ts;
        for (int r = 0; r < reps; ++r) {
            auto t0 = std::chrono::steady_clock::now();
            for (int q = 0; q < inner; ++q)
                ix->search(nq, Q.data(), k, D.data(), I.data());
            ts.push_back(std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count() / inner);
        }
        std::sort(ts.begin(), ts.end());
        char p[4096];
        snprintf(p, sizeof p, "%s_p%d.u32", pref.c_str(), P[e]);
        FILE *f = fopen(p, "wb");
        std::vector<uint32_t> out(nq * k);
        for (size_t i = 0; i < nq * k; ++i) out[i] = I[i] < 0 ? 0u : (uint32_t)I[i];
        fwrite(out.data(), 4, nq * k, f);
        fclose(f);
        printf("  {\"param\": %d, \"wall_s\": %.9f, \"n_queries\": %zu, \"inner\": %d}%s\n",
               P[e], ts[0], nq, inner, e + 1 < P.size() ? "," : "");
        fflush(stdout);
    }
    printf("]\n");
    delete ix;
    return 0;
}
