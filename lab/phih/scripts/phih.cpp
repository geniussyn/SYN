// phi_h: number of components of GT_k(q) under MUTUAL h-hop reachability
// in the directed level-0 graph.  R_h is an exact h-step BFS, h counts edges.
//
// dist(u,v) <= a+b  <=>  exists w with d_out(u,w)<=a and d_in(v,w)<=b.
// We therefore expand each GT node's out-ball to radius 2 (bit-masked by source)
// and each GT node's in-ball to radius 2, which settles h=1..4 in one sweep:
//   h=1:(a,b)=(1,0)  h=2:(1,1)  h=3:(2,1)  h=4:(2,2)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <omp.h>

static const int HMAX = 4;

static void *slurp(const char *p, size_t *sz) {
    FILE *f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", p); exit(1); }
    fseek(f, 0, SEEK_END); *sz = ftell(f); fseek(f, 0, SEEK_SET);
    void *b = malloc(*sz);
    if (fread(b, 1, *sz, f) != *sz) { fprintf(stderr, "short read %s\n", p); exit(1); }
    fclose(f); return b;
}

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: phih <prefix> <gt.i32> <k> <out.i32> [nthread]\n"); return 1; }
    const char *pre = argv[1], *gtp = argv[2], *outp = argv[4];
    const int K = atoi(argv[3]);
    const int NT = argc > 5 ? atoi(argv[5]) : 24;
    if (K > 128) { fprintf(stderr, "K<=128\n"); return 1; }
    omp_set_num_threads(NT);

    char buf[512]; size_t sz;
    snprintf(buf, sizeof buf, "%s_fwd_indptr.i64", pre);
    const int64_t *fp = (const int64_t *)slurp(buf, &sz);
    const int64_t n = (int64_t)(sz / 8) - 1;
    snprintf(buf, sizeof buf, "%s_fwd_indices.i32", pre);
    const int32_t *fi = (const int32_t *)slurp(buf, &sz);
    snprintf(buf, sizeof buf, "%s_rev_indptr.i64", pre);
    const int64_t *rp = (const int64_t *)slurp(buf, &sz);
    snprintf(buf, sizeof buf, "%s_rev_indices.i32", pre);
    const int32_t *ri = (const int32_t *)slurp(buf, &sz);

    const int32_t *gt = (const int32_t *)slurp(gtp, &sz);
    const int64_t NQ = (int64_t)(sz / 4) / K;
    fprintf(stderr, "n=%ld edges=%ld queries=%ld K=%d threads=%d\n", n, fp[n], NQ, K, NT);

    std::vector<int32_t> phi((size_t)NQ * HMAX);

    #pragma omp parallel
    {
        std::vector<uint64_t> mA1(n, 0), mA2(n, 0);
        std::vector<uint8_t>  seen(n, 0);
        std::vector<int32_t> touchedA, touchedB, cur, nxt;
        std::vector<uint8_t> Rm((size_t)HMAX * K * K);
        std::vector<uint64_t> reach((size_t)HMAX * K);
        const int NCH = (K + 63) / 64;

        #pragma omp for schedule(dynamic, 16)
        for (int64_t q = 0; q < NQ; q++) {
            const int32_t *S = gt + q * K;
            std::fill(Rm.begin(), Rm.end(), 0);

            for (int ch = 0; ch < NCH; ch++) {
                const int b0 = ch * 64, b1 = (b0 + 64 < K) ? b0 + 64 : K;

                // ---- out-balls (radius 2) for the sources in this chunk ----
                touchedA.clear();
                for (int i = b0; i < b1; i++) {
                    const uint64_t bit = (uint64_t)1 << (i - b0);
                    cur.clear();
                    int32_t s0 = S[i];
                    seen[s0] = 1; touchedB.push_back(s0);
                    if (!mA1[s0] && !mA2[s0]) touchedA.push_back(s0);
                    mA1[s0] |= bit; mA2[s0] |= bit;
                    cur.push_back(s0);
                    for (int d = 1; d <= 2; d++) {
                        nxt.clear();
                        for (int32_t u : cur)
                            for (int64_t e = fp[u]; e < fp[u + 1]; e++) {
                                int32_t v = fi[e];
                                if (seen[v]) continue;
                                seen[v] = 1; touchedB.push_back(v);
                                if (!mA1[v] && !mA2[v]) touchedA.push_back(v);
                                if (d <= 1) mA1[v] |= bit;
                                mA2[v] |= bit;
                                nxt.push_back(v);
                            }
                        cur.swap(nxt);
                    }
                    for (int32_t v : touchedB) seen[v] = 0;
                    touchedB.clear();
                }

                // ---- in-balls (radius 2) for every GT node ----
                std::fill(reach.begin(), reach.end(), 0ull);
                for (int j = 0; j < K; j++) {
                    cur.clear();
                    int32_t s0 = S[j];
                    seen[s0] = 1; touchedB.push_back(s0);
                    cur.push_back(s0);
                    reach[0 * K + j] |= mA1[s0]; reach[1 * K + j] |= mA1[s0];
                    reach[2 * K + j] |= mA2[s0]; reach[3 * K + j] |= mA2[s0];
                    for (int d = 1; d <= 2; d++) {
                        nxt.clear();
                        for (int32_t u : cur)
                            for (int64_t e = rp[u]; e < rp[u + 1]; e++) {
                                int32_t v = ri[e];
                                if (seen[v]) continue;
                                seen[v] = 1; touchedB.push_back(v);
                                if (d <= 1) { reach[1 * K + j] |= mA1[v]; reach[2 * K + j] |= mA2[v]; }
                                reach[3 * K + j] |= mA2[v];
                                nxt.push_back(v);
                            }
                        cur.swap(nxt);
                    }
                    for (int32_t v : touchedB) seen[v] = 0;
                    touchedB.clear();
                }
                for (int32_t v : touchedA) { mA1[v] = 0; mA2[v] = 0; }

                for (int h = 0; h < HMAX; h++)
                    for (int j = 0; j < K; j++) {
                        uint64_t m = reach[(size_t)h * K + j];
                        while (m) { int t = __builtin_ctzll(m); m &= m - 1;
                                    Rm[((size_t)h * K + (b0 + t)) * K + j] = 1; }
                    }
            }

            // ---- components under mutual reachability ----
            std::vector<int> par(K);
            for (int h = 0; h < HMAX; h++) {
                for (int i = 0; i < K; i++) par[i] = i;
                auto find = [&](int x) { while (par[x] != x) { par[x] = par[par[x]]; x = par[x]; } return x; };
                for (int i = 0; i < K; i++)
                    for (int j = i + 1; j < K; j++)
                        if (Rm[((size_t)h * K + i) * K + j] && Rm[((size_t)h * K + j) * K + i]) {
                            int a = find(i), b = find(j); if (a != b) par[a] = b;
                        }
                int c = 0;
                for (int i = 0; i < K; i++) if (find(i) == i) c++;
                phi[(size_t)q * HMAX + h] = c;
            }
        }
    }

    FILE *f = fopen(outp, "wb");
    fwrite(phi.data(), 4, phi.size(), f);
    fclose(f);
    fprintf(stderr, "wrote %s\n", outp);
    return 0;
}
