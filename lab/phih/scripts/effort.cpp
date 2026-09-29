// Per-query effort: NDC (distance computations) of an ef-search over the stored
// level-0 graph, swept over an ef ladder.  The y-axis Steiner-hardness and DARTH
// are actually built to predict: least NDC to reach a target recall.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <queue>
#include <algorithm>
#include <omp.h>

static void *slurp(const char *p, size_t *sz) {
    FILE *f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", p); exit(1); }
    fseek(f, 0, SEEK_END); *sz = ftell(f); fseek(f, 0, SEEK_SET);
    void *b = malloc(*sz);
    if (fread(b, 1, *sz, f) != *sz) { fprintf(stderr, "short read %s\n", p); exit(1); }
    fclose(f); return b;
}

int main(int argc, char **argv) {
    if (argc < 9) {
        fprintf(stderr, "usage: effort <prefix> <base.f32> <query.f32> <dim> <gt.i32> <k> "
                        "<entry> <metric l2|cos> <out.i64> [nthread] [ef...]\n");
        return 1;
    }
    const char *pre = argv[1];
    size_t sz;
    char buf[512];
    snprintf(buf, sizeof buf, "%s_fwd_indptr.i64", pre);
    const int64_t *ip = (const int64_t *)slurp(buf, &sz);
    const int64_t n = (int64_t)(sz / 8) - 1;
    snprintf(buf, sizeof buf, "%s_fwd_indices.i32", pre);
    const int32_t *ix = (const int32_t *)slurp(buf, &sz);

    const float *base  = (const float *)slurp(argv[2], &sz);
    const float *query = (const float *)slurp(argv[3], &sz);
    const int64_t dim  = atoll(argv[4]);
    const int64_t NQ   = (int64_t)(sz / 4) / dim;
    const int32_t *gt  = (const int32_t *)slurp(argv[5], &sz);
    const int K        = atoi(argv[6]);
    const int32_t ep   = atoi(argv[7]);
    const bool l2      = strcmp(argv[8], "l2") == 0;
    const char *outp   = argv[9];
    const int NT       = argc > 10 ? atoi(argv[10]) : 24;
    std::vector<int> EFS;
    for (int i = 11; i < argc; i++) EFS.push_back(atoi(argv[i]));
    if (EFS.empty()) EFS = {10, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048};
    omp_set_num_threads(NT);
    fprintf(stderr, "n=%ld dim=%ld queries=%ld K=%d ef-ladder=%zu entry=%d %s\n",
            n, dim, NQ, K, EFS.size(), ep, l2 ? "l2" : "cos");

    // out[q][e] = ndc, and hits[q][e]
    std::vector<int64_t> ndc((size_t)NQ * EFS.size());
    std::vector<int64_t> hit((size_t)NQ * EFS.size());

    #pragma omp parallel
    {
        std::vector<uint32_t> stamp(n, 0);
        uint32_t gen = 0;
        #pragma omp for schedule(dynamic, 4)
        for (int64_t q = 0; q < NQ; q++) {
            const float *qv = query + q * dim;
            for (size_t ei = 0; ei < EFS.size(); ei++) {
                const int ef = EFS[ei];
                gen++;
                int64_t nd = 0;
                auto dist = [&](int32_t u) {
                    const float *x = base + (int64_t)u * dim;
                    float s = 0;
                    if (l2) { for (int64_t i = 0; i < dim; i++) { float d = x[i] - qv[i]; s += d * d; } }
                    else    { for (int64_t i = 0; i < dim; i++) s += x[i] * qv[i]; s = 1.0f - s; }
                    nd++;
                    return s;
                };
                typedef std::pair<float, int32_t> P;
                std::priority_queue<P, std::vector<P>, std::greater<P>> cand;  // min-heap
                std::priority_queue<P> top;                                    // max-heap
                float d0 = dist(ep);
                cand.push({d0, ep}); top.push({d0, ep});
                stamp[ep] = gen;
                while (!cand.empty()) {
                    auto [dc, u] = cand.top();
                    if (dc > top.top().first && (int)top.size() >= ef) break;
                    cand.pop();
                    for (int64_t e = ip[u]; e < ip[u + 1]; e++) {
                        int32_t v = ix[e];
                        if (stamp[v] == gen) continue;
                        stamp[v] = gen;
                        float dv = dist(v);
                        if ((int)top.size() < ef || dv < top.top().first) {
                            cand.push({dv, v}); top.push({dv, v});
                            if ((int)top.size() > ef) top.pop();
                        }
                    }
                }
                std::vector<int32_t> res;
                while (!top.empty()) { res.push_back(top.top().second); top.pop(); }
                std::reverse(res.begin(), res.end());          // nearest first
                if ((int)res.size() > K) res.resize(K);
                std::sort(res.begin(), res.end());
                int h = 0;
                for (int i = 0; i < K; i++)
                    if (std::binary_search(res.begin(), res.end(), gt[q * K + i])) h++;
                ndc[(size_t)q * EFS.size() + ei] = nd;
                hit[(size_t)q * EFS.size() + ei] = h;
            }
        }
    }
    FILE *f = fopen(outp, "wb");
    int64_t meta[3] = {NQ, (int64_t)EFS.size(), K};
    fwrite(meta, 8, 3, f);
    std::vector<int64_t> efs(EFS.begin(), EFS.end());
    fwrite(efs.data(), 8, efs.size(), f);
    fwrite(ndc.data(), 8, ndc.size(), f);
    fwrite(hit.data(), 8, hit.size(), f);
    fclose(f);
    fprintf(stderr, "wrote %s\n", outp);
    return 0;
}
