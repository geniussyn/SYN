"""Agreement, final shares, and the join back onto the k-means labels.

Reads the two annotators' records, reports what the paper has to show:

  1. final human shares with 95% Wilson intervals
  2. pre-adjudication label agreement -- Cohen's kappa and Krippendorff's alpha
  3. grouping agreement -- pairwise F1 and ARI over the 45 slot pairs per query
  4. the confusion matrix of human label against the k-means heuristic label

and the per-label retrieval columns (phi_1, recall@ef16, largest semantic-group
fraction) that replace the old taxonomy table.

phi_3 is available behind --phi3 and is deliberately not in the paper's table.
phi_1 and the recall column are read from frag_types.npz, which the campaign
that produced the labels wrote; phi_3 is recomputed from a stored graph, and a
graph is a function of the build. Across independent builds of the same
embedding the per-label ordering of phi_3 does not hold, while phi_1 is
unchanged. Reporting a column that moves with the build under a taxonomy that
does not would be reporting the builder.

The heuristic label is joined LAST, from key_all.jsonl. Nothing in this file
feeds back into the guideline -- it is run after the labels are frozen.

  NOTE the two kappas are different things. Cohen's kappa here is inter-annotator
  agreement. It is not the cohesion kappa that was dropped from the paper.
"""
import argparse, csv, json, math, sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import numpy as np
from paths import ROOT, RESULTS  # noqa: E402

LAB = ROOT; RES = LAB / "results"; ANN = LAB / "studies" / "annotation"
LABELS = ("single_topic", "genuine_polysemy", "surface_collision", "mixed", "unclear")
# the k-means rule's four names, in the order frag_types.py emits them
HEUR = ("coherent", "polysemous", "loose senses", "scattered")
HEUR_AS = {"coherent": "single topic", "polysemous": "polysemy",
           "loose senses": "weak senses", "scattered": "surface collision"}


# ---------------------------------------------------------------- io
def load_records(path):
    """One annotator's records, keyed by query_id. Accepts .jsonl or .csv."""
    p = Path(path); rows = []
    if p.suffix == ".jsonl":
        rows = [json.loads(l) for l in p.read_text().split("\n") if l.strip()]
    else:
        rows = list(csv.DictReader(p.open(encoding="utf-8")))
    out = {}
    for r in rows:
        lab = (r.get("primary_label") or "").strip()
        if not lab:
            continue                                    # unannotated sheet line
        if lab not in LABELS:
            sys.exit(f"{p}: query {r.get('query_id')} has unknown label {lab!r}")
        g = [x.strip() for x in str(r.get("semantic_group_ids", "")).split(",")]
        g = [x for x in g if x]
        if len(g) != 10:
            sys.exit(f"{p}: query {r.get('query_id')} has {len(g)} group ids, need 10")
        if not (r.get("brief_rationale") or "").strip():
            print(f"  warn: {p.name} query {r.get('query_id')} has no brief_rationale")
        out[str(r["query_id"])] = {
            "label": lab, "groups": g,
            "confidence": (r.get("confidence") or "").strip(),
            "rationale": (r.get("brief_rationale") or "").strip(),
            "guideline_version": (r.get("guideline_version") or "").strip(),
        }
    return out


# ---------------------------------------------------------------- statistics
def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def cohen_kappa(a, b, cats):
    n = len(a)
    obs = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    exp = sum((ca[c] / n) * (cb[c] / n) for c in cats)
    return obs, (obs - exp) / (1 - exp) if exp < 1 else float("nan")


def krippendorff_alpha(units, cats):
    """Nominal alpha. units: list of lists of values (one per coder present)."""
    o = defaultdict(float)
    for vals in units:
        m = len(vals)
        if m < 2:
            continue
        for i in range(m):
            for j in range(m):
                if i != j:
                    o[(vals[i], vals[j])] += 1.0 / (m - 1)
    nrow = {c: sum(v for (x, _), v in o.items() if x == c) for c in cats}
    n = sum(nrow.values())
    if n < 2:
        return float("nan")
    Do = sum(v for (x, y), v in o.items() if x != y) / n
    De = sum(nrow[x] * nrow[y] for x in cats for y in cats if x != y) / (n * (n - 1))
    return 1 - Do / De if De > 0 else float("nan")


def pair_labels(groups, uncertain="exclude"):
    """The 45 slot pairs -> True if the two slots share a group. None = skip."""
    out = {}
    for i, j in combinations(range(10), 2):
        gi, gj = groups[i], groups[j]
        if uncertain == "exclude" and ("U" in (gi.upper(), gj.upper())):
            out[(i, j)] = None
        elif gi.upper() == "U" or gj.upper() == "U":
            out[(i, j)] = False                    # 'singleton': U is its own group
        else:
            out[(i, j)] = gi == gj
    return out


def pairwise_prf(pa, pb):
    """tp, fp, fn of 'same group' with annotator A as the reference."""
    tp = fp = fn = 0
    for k in pa:
        x, y = pa[k], pb[k]
        if x is None or y is None:
            continue
        tp += x and y
        fp += (not x) and y
        fn += x and (not y)
    return tp, fp, fn


def ari(ga, gb):
    """Adjusted Rand index of two groupings of the same 10 slots."""
    tab = Counter(zip(ga, gb))
    a = Counter(ga); b = Counter(gb); n = len(ga)
    c2 = lambda x: x * (x - 1) / 2
    idx = sum(c2(v) for v in tab.values())
    ea = sum(c2(v) for v in a.values()); eb = sum(c2(v) for v in b.values())
    exp = ea * eb / c2(n); mx = (ea + eb) / 2
    return (idx - exp) / (mx - exp) if mx != exp else 1.0


# ---------------------------------------------------------------- report
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a1", default=str(ANN / "records_main_M1.jsonl"))
    ap.add_argument("--a2", default=str(ANN / "records_main_A2.jsonl"))
    ap.add_argument("--adjudicated", default=str(ANN / "records_adjudicated.jsonl"),
                    help="third annotator / consensus, disagreements only")
    ap.add_argument("--key", default=str(ANN / "key_all.jsonl"))
    ap.add_argument("--uncertain", choices=("exclude", "singleton"), default="exclude")
    ap.add_argument("--phi3", action="store_true",
                    help="also report phi_3 per label. NOT in the paper: phi_3 "
                         "is recomputed from a graph, and the per-label "
                         "separation it shows does not survive a rebuild of the "
                         "same embedding (see results/build_stability_*.json), "
                         "whereas phi_1 does. Kept for inspection.")
    ap.add_argument("--single", action="store_true", default=True,
                    help="one annotator only (--a1): report shares and structure, "
                         "skip the agreement sections, which need a second pass")
    ap.add_argument("--out", default=str(RES / "annot_summary.json"))
    a = ap.parse_args()

    A1 = load_records(a.a1)
    A2 = {} if a.single else load_records(a.a2)
    # key_all.jsonl carries the row index into the hard decile and the k-means
    # heuristic label, and is what section 4's confusion matrix compares against.
    # It is not needed for the taxonomy table, so when it is absent the row index
    # is rebuilt from the item file and section 4 is skipped -- which is what
    # lets this run from the shipped annotation alone.
    kp = Path(a.key)
    if kp.exists():
        key = {str(r["query_id"]): r for r in
               map(json.loads, filter(None, kp.read_text().split("\n")))}
    else:
        items = [json.loads(l) for l in
                 (ANN / "items_all.jsonl").read_text().splitlines() if l.strip()]
        key = {str(r["query_id"]): {"row": i} for i, r in enumerate(items)}
        print(f"  {kp.name} absent; row index rebuilt from items_all.jsonl "
              f"({len(key)} queries), heuristic confusion skipped")
    adj = {}
    if Path(a.adjudicated).exists():
        adj = load_records(a.adjudicated)

    both = sorted(A1 if a.single else set(A1) & set(A2), key=lambda q: key[q]["row"])
    if a.single:
        print(f"single-annotator mode: {len(both)} / {len(key)} annotated by {a.a1}")
        print("  no second pass -> no kappa/alpha, no grouping agreement; "
              "shares below are one annotator's labels, not adjudicated human consensus")
    else:
        print(f"annotated by both: {len(both)} / {len(key)}"
              f"   (A1 {len(A1)}, A2 {len(A2)}, adjudicated {len(adj)})")
    gv = {r["guideline_version"] for r in list(A1.values()) + list(A2.values())}
    print(f"guideline_version(s): {sorted(gv)}")
    if len(gv) > 1:
        print("  warn: records span more than one guideline version")

    # ---- 2. pre-adjudication label agreement
    obs = ck = al = micro = None
    f1s = aris = []
    if a.single:
        print("\n[2] [3] skipped: agreement statistics require two annotators")
    else:
      la = [A1[q]["label"] for q in both]; lb = [A2[q]["label"] for q in both]
      obs, ck = cohen_kappa(la, lb, LABELS)
      al = krippendorff_alpha([[A1[q]["label"], A2[q]["label"]] for q in both], LABELS)
      print(f"\n[2] label agreement, before adjudication")
      print(f"    raw agreement      {obs:.3f}  ({sum(x==y for x,y in zip(la,lb))}/{len(both)})")
      print(f"    Cohen's kappa      {ck:.3f}   (inter-annotator; not the cohesion kappa)")
      print(f"    Krippendorff alpha {al:.3f}")

      # ---- 3. grouping agreement
      f1s, aris = [], []; TP = FP = FN = 0
      for q in both:
        pa = pair_labels(A1[q]["groups"], a.uncertain)
        pb = pair_labels(A2[q]["groups"], a.uncertain)
        tp, fp, fn = pairwise_prf(pa, pb)
        TP += tp; FP += fp; FN += fn
        f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0)
        aris.append(ari(A1[q]["groups"], A2[q]["groups"]))
      micro = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) else 1.0
      print(f"\n[3] semantic-group agreement  (uncertain: {a.uncertain})")
      print(f"    pairwise F1  macro {np.mean(f1s):.3f}   micro {micro:.3f}"
          f"   ({TP} same-pairs agreed, {FP} + {FN} not)")
      print(f"    ARI          macro {np.mean(aris):.3f} +- {np.std(aris):.3f}")

    # ---- final label
    final, unresolved = {}, []
    for q in both:
        if a.single:
            final[q] = A1[q]["label"]
        elif q in adj:
            final[q] = adj[q]["label"]
        elif A1[q]["label"] == A2[q]["label"]:
            final[q] = A1[q]["label"]
        else:
            unresolved.append(q)
    if unresolved:
        adjpath = Path(a.a1).parent / "to_adjudicate.jsonl"
        print(f"\n  {len(unresolved)} disagreements still need adjudication -> {adjpath}")
        rows = [{"query_id": q, "row": key[q]["row"],
                 "A1": A1[q]["label"], "A2": A2[q]["label"],
                 "A1_rationale": A1[q]["rationale"], "A2_rationale": A2[q]["rationale"]}
                for q in unresolved]
        adjpath.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))

    # ---- 1. final shares
    n = len(final); cnt = Counter(final.values())
    print(f"\n[1] final human shares, n={n}")
    print(f"    {'label':20}{'n':>5}{'share':>8}{'95% CI':>18}")
    shares = {}
    for L in LABELS:
        lo, hi = wilson(cnt[L], n)
        shares[L] = {"n": cnt[L], "share": cnt[L] / n if n else 0.0, "ci": [lo, hi]}
        print(f"    {L:20}{cnt[L]:5}{cnt[L]/n if n else 0:8.1%}"
              f"{f'[{lo:.1%}, {hi:.1%}]':>18}")

    # ---- per-label retrieval columns
    rowsof = lambda qs: np.array([key[q]["row"] for q in qs], dtype=int)
    ft = np.load(RES / "frag_types.npz", allow_pickle=True)
    phi1, rec = ft["phi1"].astype(float), ft["recall"].astype(float)
    phi3 = None
    if a.phi3:
        sys.path.insert(0, str(LAB / "code"))
        from kappa import phi                                       # noqa: E402
        # the campaign that produced these labels wrote its populations under an
        # "ord100" tag; the current one writes "own_efc200" for the same cell
        # (M=16, efC=200, baseline arm). Take whichever is present.
        cand = [f"msmarco-1m__vanilla__M16__{t}" for t in ("ord100", "own_efc200")]
        stem = next((c for c in cand if (RES / f"pops_{c}.npz").exists()
                     and (RES / f"graph_{c}.npz").exists()), None)
        if stem is None:
            raise SystemExit(
                "--phi3 needs the hard-decile population and the level-0 graph "
                "for MS MARCO / vanilla / M=16, efC=200. Looked for "
                + ", ".join(f"pops_{c}.npz" for c in cand))
        hard = np.load(RES / f"pops_{stem}.npz")["hard"]
        T = np.load(RES / "gt_msmarco-1m__vanilla_k10.npz")["idx"][hard].astype(np.int64)
        g = np.load(RES / f"graph_{stem}.npz")
        phi3 = phi(g["indptr"], g["indices"], T, h=3, log_every=0).astype(float)

    def lgf(q):
        """largest semantic-group fraction, from the grouping that decided the label."""
        src = adj.get(q) or A1[q]
        g = [x for x in src["groups"] if x.upper() != "U"]
        return max(Counter(g).values()) / 10 if g else float("nan")

    print(f"\n[4] per-label retrieval structure")
    hdr = f"    {'label':20}{'n':>5}{'phi_1':>8}"
    hdr += f"{'phi_3':>8}" if phi3 is not None else ""
    print(hdr + f"{'recall@ef16':>13}{'largest grp':>13}")
    percat = {}
    for L in LABELS:
        qs = [q for q in final if final[q] == L]
        if not qs:
            continue
        r = rowsof(qs); f = np.nanmean([lgf(q) for q in qs])
        line = f"    {L:20}{len(qs):5}{phi1[r].mean():8.2f}"
        line += f"{phi3[r].mean():8.2f}" if phi3 is not None else ""
        print(line + f"{rec[r].mean():13.3f}{f:13.2f}")
        percat[L] = {"n": len(qs), "phi1": float(phi1[r].mean()),
                     "phi3": float(phi3[r].mean()) if phi3 is not None else None,
                     "recall_ef16": float(rec[r].mean()),
                     "largest_group_fraction": float(f)}

    # ---- 5. confusion against the k-means heuristic  (joined last)
    print(f"\n[5] human label x k-means heuristic label (k=2048)")
    print(f"    {'':20}" + "".join(f"{HEUR_AS[h]:>19}" for h in HEUR))
    conf = {}
    for L in LABELS:
        qs = [q for q in final if final[q] == L]
        if not all("heuristic_kind" in key[q] for q in qs):
            print("    skipped: no heuristic label in the key file")
            break
        c = Counter(key[q]["heuristic_kind"] for q in qs)
        conf[L] = {h: c[h] for h in HEUR}
        if qs:
            print(f"    {L:20}" + "".join(f"{c[h]:>19}" for h in HEUR))
    if conf:
        agree = sum(conf[L][h] for L, h in
                    (("single_topic", "coherent"), ("genuine_polysemy", "polysemous"),
                     ("surface_collision", "scattered")))
        print(f"\n    heuristic reproduces the human label on {agree}/{n} = "
              f"{agree/n if n else 0:.1%} of the queries it has a counterpart for"
              f"  (no heuristic counterpart to mixed/unclear; 'weak senses' is retired)")
    else:
        agree = None

    Path(a.out).write_text(json.dumps({
        "n_both": len(both), "n_final": n, "guideline_versions": sorted(gv),
        "heuristic_agreement": agree,
        "uncertain_mode": a.uncertain,
        "shares": shares,
        "agreement": None if a.single else {
            "raw": obs, "cohen_kappa": ck, "krippendorff_alpha": al,
            "pairwise_f1_macro": float(np.mean(f1s)),
            "pairwise_f1_micro": micro,
            "ari_macro": float(np.mean(aris)), "ari_sd": float(np.std(aris))},
        "annotators": [a.a1] if a.single else [a.a1, a.a2],
        "single_annotator": bool(a.single),
        "per_label": percat, "confusion_vs_kmeans": conf,
        "n_unresolved": len(unresolved),
    }, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
