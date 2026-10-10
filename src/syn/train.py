"""Stage 2 -- SYN: InfoNCE with region negatives and a boundary-aware pull.

    L = L_nce + lambda * L_pull

L_nce is the base objective, with the positive drawn from S_q under the frozen
score law and the rest of S_q masked out of the denominator: drawing the
positive uniformly and leaving the other K-1 anchors as negatives trains the
model to separate the documents the method exists to join.

L_pull acts only on A_q, against each document's own boundary band:

    1/2 [ softplus((s(d_i,b_i) - s(d_i,d_j) + m) / tau_b)
        + softplus((s(d_j,b_j) - s(d_j,d_i) + m) / tau_b) ]

Its gradient is restricted to the query-radial tangent. The operator's forward
pass is the identity, so the loss above is what is optimised either way; what
changes is that a document is moved along the direction that alters its query
similarity rather than along the whole document-to-document direction, which
would also rotate it against every other document in its neighbourhood. Measured
on MS MARCO 1M against the unrestricted gradient, it gives up 5.7% of the
fragmentation gain and recovers 40% of the exhaustive ranking the intervention
otherwise costs.

Stage 3 is line 1 of the epilogue: merge the adapter, re-encode the corpus,
rebuild the index. The search algorithm is untouched.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from syn.corpus import load_corpus                                   # noqa: E402
from syn.encoders import encode, forward_embed, load_encoder, \
    lora_targets, spec_record                                        # noqa: E402
from syn.env import ROOT, log, provenance, write_json                # noqa: E402


def qradial(z, q):
    """Restrict the pull gradient to the query-radial tangent (Appendix A).

    For a unit z with a = q.z, write z = a q + rho u with rho = sqrt(1 - a^2)
    and u perpendicular to q. The unit tangent t = rho q - a u is the direction
    that changes q.z to first order while holding the azimuth fixed. This is a
    straight-through operator: the forward pass is the identity, so the loss
    value is unchanged, and the Jacobian is t t^T, so only that component of the
    gradient survives. The query is held fixed.
    """
    q = q.detach()
    a = (z * q).sum(-1, keepdim=True)
    rho = (1.0 - a * a).clamp(min=1e-12).sqrt()
    u = (z - a * q) / rho
    t = rho * q - a * u
    t = t / t.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    # t is a direction, not a function of z for the purpose of the Jacobian:
    # leaving it attached would add a second path and the operator would no
    # longer be the projector t t^T.
    t = t.detach()
    perp = z - (z * t).sum(-1, keepdim=True) * t
    return z - perp + perp.detach()


class Regions:
    """Inverted index over the k-means assignment: C(v) in O(1)."""

    def __init__(self, assign, n_clusters):
        self.assign = assign
        self.order = np.argsort(assign, kind="stable").astype(np.int32)
        self.start = np.searchsorted(assign[self.order], np.arange(n_clusters + 1))

    def members(self, doc):
        c = self.assign[doc]
        return self.order[self.start[c]:self.start[c + 1]]

    def sample_global(self, doc, n, forbid, rng):
        """n documents drawn uniformly from the whole corpus, not the region.

        The ablation of cluster negatives. `doc` is ignored, which is the
        point: a region negative is near the anchor and so is a hard negative
        the InfoNCE term has to work to separate, while a global one is almost
        always far and separates itself.
        """
        N = len(self.assign)
        out, seen = [], set(forbid)
        while len(out) < n:
            for d in rng.integers(0, N, size=2 * n):
                d = int(d)
                if d not in seen:
                    seen.add(d)
                    out.append(d)
                    if len(out) == n:
                        break
        return np.asarray(out, dtype=np.int32)

    def sample(self, doc, n, forbid, rng):
        """n documents drawn uniformly from C(doc) \\ forbid."""
        pool = self.members(doc)
        if pool.size <= n + len(forbid):
            cand = pool[~np.isin(pool, list(forbid))]
            if cand.size == 0:
                return rng.choice(len(self.assign), n, replace=False).astype(np.int32)
            return rng.choice(cand, n, replace=cand.size < n).astype(np.int32)
        out, seen = [], set(forbid)
        draw = min(2 * n, pool.size)
        while len(out) < n:
            for d in rng.choice(pool, draw, replace=False):
                d = int(d)
                if d not in seen:
                    seen.add(d)
                    out.append(d)
                    if len(out) == n:
                        break
        return np.asarray(out, dtype=np.int32)


@torch.no_grad()
def monitor_stats(mdl, tok, spec, pool_texts, mon_texts, S_mon, bs, max_len,
                  precision, deg=22):
    """Watched, not measured. phi needs a 1M index, so the same direction is
    read inside a fixed reference pool, in the quantities the paper defines:

      median_mutual_rank   rank_u(v) among the frozen targets, median per query
                           and geometric mean over queries -- Figure 2's
                           quantity. Should fall.
      pair_within_degree   share of target pairs whose mutual rank is within one
                           out-degree, so an edge between them is possible.
                           Should rise; it is phi_1's direction.
      target_overlap       how much of the frozen target set is still in the
                           pool's own top-K. The cost side: the objective is
                           free to replace targets, and it does.
    """
    was_training = mdl.training
    mdl.eval()
    P, _ = encode(mdl, tok, spec, pool_texts, is_query=False, bs=bs,
                  max_len=max_len, log_every=0, precision=precision)
    Qm, _ = encode(mdl, tok, spec, mon_texts, is_query=True, bs=bs,
                   max_len=max_len, log_every=0, precision=precision)
    if was_training:
        mdl.train()
    Pg = torch.from_numpy(P).cuda()
    K = S_mon.shape[1]
    top = torch.topk(torch.from_numpy(Qm).cuda() @ Pg.T, K, dim=1).indices.cpu().numpy()
    overlap = float(np.mean([len(set(t) & set(r)) / K for t, r in zip(top, S_mon)]))
    gt = np.unique(S_mon)
    row_of = {int(g): i for i, g in enumerate(gt)}
    Sim = Pg[torch.from_numpy(gt.astype(np.int64)).cuda()] @ Pg.T
    med, within = [], []
    for row in S_mon:
        u = torch.tensor([row_of[int(x)] for x in row], device="cuda")
        cols = torch.from_numpy(row.astype(np.int64)).cuda()
        sub = Sim[u]
        thr = sub[torch.arange(len(row), device="cuda")[:, None], cols[None, :]]
        rk = (sub[:, None, :] > thr[:, :, None]).sum(-1)
        off = ~torch.eye(len(row), dtype=torch.bool, device="cuda")
        r = rk[off].float().clamp(min=1)
        med.append(float(r.median()))
        within.append(float((r <= deg).float().mean()))
    del Pg, Sim
    torch.cuda.empty_cache()
    return {"median_mutual_rank": float(np.exp(np.mean(np.log(med)))),
            "pair_within_degree": float(np.mean(within)),
            "target_overlap": overlap}


def main():
    ap = argparse.ArgumentParser(description="Stage 2: SYN")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--corpora-dir", default=str(ROOT / "corpora"))
    ap.add_argument("--mine-dir", required=True)
    ap.add_argument("--pull-name", default="pull",
                    help="which pull file in --mine-dir to train against")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default="syn")
    ap.add_argument("--encoder", default="bge-m3")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch-q", type=int, default=2)
    ap.add_argument("--n-neg", type=int, default=4, help="negatives per anchor")
    ap.add_argument("--neg-source", choices=("region", "global"), default="region",
                    help="where InfoNCE negatives come from. region is the "
                         "query's k-means cell, the shipped setting; global "
                         "draws from the whole corpus and is the ablation.")
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--lambda-pull", type=float, default=1.0)
    ap.add_argument("--pull-tau", type=float, default=0.05, help="tau_b")
    ap.add_argument("--pull-margin", type=float, default=0.0, help="m")
    ap.add_argument("--pull-grad", choices=["qradial", "plain"], default="qradial",
                    help="qradial restricts the pull gradient to the "
                         "query-radial tangent; plain lets it use the whole "
                         "document-to-document direction. The loss value is "
                         "identical either way, so this is the gradient and "
                         "nothing else.")
    ap.add_argument("--pos-select", choices=["softmax", "uniform"], default="softmax")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--schedule", choices=["onecycle", "none"], default="onecycle")
    ap.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--no-grad-checkpoint", action="store_true")
    ap.add_argument("--seed", type=int, default=20260808)
    ap.add_argument("--monitor-every", type=int, default=100)
    ap.add_argument("--monitor-bs", type=int, default=128)
    ap.add_argument("--allow-remote-code", action="store_true")
    a = ap.parse_args()

    src, out = Path(a.mine_dir), Path(a.out_dir) / f"adapter_{a.tag}"
    D = np.load(src / "mine.npz", allow_pickle=True)
    S_train, assign = D["S_train"], D["assign"]
    n_clusters = int(D["n_clusters"])
    P = np.load(src / f"{a.pull_name}.npz", allow_pickle=True)
    pos_prob, band, anchors = P["pos_prob"], P["band"], P["anchors"]
    pair_q, pair_i, pair_j = P["pair_q"], P["pair_i"], P["pair_j"]
    anchor_row = {int(d): r for r, d in enumerate(anchors)}
    pairs_of = [[] for _ in range(len(S_train))]
    for q, i, j in zip(pair_q, pair_i, pair_j):
        pairs_of[int(q)].append((int(i), int(j)))

    qtext = [json.loads(l)["text"] for l in
             (src / "queries_train.jsonl").read_text(encoding="utf-8").split("\n")
             if l.strip()]
    docs, *_ = load_corpus(a.corpus, Path(a.corpora_dir))
    nq, K = S_train.shape
    max_len = a.max_len or (256 if a.corpus.startswith("msmarco") else None)

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    regions = Regions(assign, n_clusters)
    neg_sample = (regions.sample if a.neg_source == "region"
                  else regions.sample_global)
    log(f"SYN {a.corpus} | {nq:,} queries, K={K}, {a.steps} steps, "
        f"batch {a.batch_q}q, lambda {a.lambda_pull}, {a.precision}")
    log(f"active pairs available on {sum(bool(p) for p in pairs_of):,}/{nq:,} queries")

    from peft import LoraConfig, get_peft_model
    tok, mdl, spec = load_encoder(a.encoder, dtype=torch.float32,
                                  allow_remote_code=a.allow_remote_code)
    targets = lora_targets(spec)
    mdl = get_peft_model(mdl, LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r,
                                         lora_dropout=0.05, bias="none",
                                         target_modules=targets))
    if not a.no_grad_checkpoint:
        mdl.enable_input_require_grads()
        mdl.gradient_checkpointing_enable()
    trainable = [p for p in mdl.parameters() if p.requires_grad]
    ntr = sum(p.numel() for p in trainable)
    tot = sum(p.numel() for p in mdl.parameters())
    if ntr == 0:
        raise SystemExit(f"no LoRA modules matched {targets}")
    log(f"LoRA r={a.lora_r} on {targets}: {ntr:,}/{tot:,} trainable ({ntr / tot:.3%})")

    opt = torch.optim.Adam(trainable, lr=a.lr)
    sched = (torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr,
                                                 total_steps=a.steps, pct_start=0.1)
             if a.schedule == "onecycle" else None)

    mon = None
    if a.monitor_every and (src / "monitor.npz").exists():
        M = np.load(src / "monitor.npz", allow_pickle=True)
        mon_text = [json.loads(l)["text"] for l in
                    (src / "queries_monitor.jsonl").read_text(encoding="utf-8")
                    .split("\n") if l.strip()]
        mon = ([docs[int(i)] for i in M["pool"]], mon_text, M["S_mon"])
        log(f"monitor: {len(mon_text):,} queries over {len(mon[0]):,} documents, "
            f"every {a.monitor_every} steps")

    hist, monlog = [], []
    if mon is not None:
        m0 = monitor_stats(mdl, tok, spec, *mon, a.monitor_bs, max_len, a.precision)
        m0["step"] = -1
        monlog.append(m0)
        log(f"  [monitor] step   -1  mutual rank {m0['median_mutual_rank']:.2f}  "
            f"within degree {m0['pair_within_degree']:.3f}  "
            f"target overlap {m0['target_overlap']:.3f}")

    mdl.train()
    t0, n_neg_seen, n_pull_steps = time.time(), [], 0
    for step in range(a.steps):
        qsel = rng.choice(nq, a.batch_q, replace=False)
        doc_ids, slot = [], {}

        def add(d):
            d = int(d)
            if d not in slot:
                slot[d] = len(doc_ids)
                doc_ids.append(d)
            return slot[d]

        q_texts, per_q = [], []
        for qi in qsel:
            qi = int(qi)
            S = S_train[qi]
            Sset = {int(x) for x in S}
            if a.pos_select == "softmax":
                # Renormalise in float64: the stored law is float32, and a
                # float32 denominator leaves a residual above numpy's tolerance.
                law = pos_prob[qi].astype(np.float64)
                pos = int(rng.choice(K, p=law / law.sum()))
            else:
                pos = int(rng.integers(K))
            s_slots = [add(v) for v in S]
            for v in S:
                for z in neg_sample(int(v), a.n_neg, Sset, rng):
                    add(z)
            pairs = pairs_of[qi]
            b_slot = {}
            for u in {S[i] for i, _ in pairs} | {S[j] for _, j in pairs}:
                b_slot[int(u)] = add(rng.choice(band[anchor_row[int(u)]]))
            q_texts.append(qtext[qi])
            per_q.append({"S": s_slots, "pos": s_slots[pos], "pairs": pairs,
                          "b": b_slot, "ids": S})

        Vq = forward_embed(mdl, tok, spec, q_texts, True, max_len=max_len,
                           precision=a.precision)
        Vd = forward_embed(mdl, tok, spec, [docs[d] for d in doc_ids], False,
                           max_len=max_len, precision=a.precision)
        logits = (Vq @ Vd.T) / a.tau

        l_nce = 0.0
        for bi, rec in enumerate(per_q):
            keep = torch.ones(Vd.shape[0], dtype=torch.bool, device=Vd.device)
            keep[rec["S"]] = False
            keep[rec["pos"]] = True
            idx = torch.nonzero(keep, as_tuple=True)[0]
            tgt = (idx == rec["pos"]).nonzero(as_tuple=True)[0]
            l_nce = l_nce + F.cross_entropy(logits[bi, idx][None, :], tgt)
            n_neg_seen.append(int(idx.numel()) - 1)
        l_nce = l_nce / len(per_q)

        di, dj, bi_, bj_, owner = [], [], [], [], []
        for bi, rec in enumerate(per_q):
            for i, j in rec["pairs"]:
                u, v = int(rec["ids"][i]), int(rec["ids"][j])
                di.append(rec["S"][i])
                dj.append(rec["S"][j])
                bi_.append(rec["b"][u])
                bj_.append(rec["b"][v])
                owner.append(bi)
        if di:
            di, dj = torch.tensor(di, device=Vd.device), torch.tensor(dj, device=Vd.device)
            bi_, bj_ = torch.tensor(bi_, device=Vd.device), torch.tensor(bj_, device=Vd.device)
            owner = torch.tensor(owner, device=Vd.device)
            zi, zj = Vd[di], Vd[dj]
            zbi, zbj = Vd[bi_], Vd[bj_]
            if a.pull_grad == "qradial":
                qv = Vq[owner]
                zi, zj = qradial(zi, qv), qradial(zj, qv)
                zbi, zbj = qradial(zbi, qv), qradial(zbj, qv)
            s_ij = (zi * zj).sum(-1)
            term = 0.5 * (F.softplus(((zi * zbi).sum(-1) - s_ij
                                      + a.pull_margin) / a.pull_tau)
                          + F.softplus(((zj * zbj).sum(-1) - s_ij
                                        + a.pull_margin) / a.pull_tau))
            cnt = torch.zeros(len(per_q), device=Vd.device).index_add_(
                0, owner, torch.ones_like(term))
            tot_q = torch.zeros(len(per_q), device=Vd.device).index_add_(0, owner, term)
            l_pull = (tot_q[cnt > 0] / cnt[cnt > 0]).mean()
            loss = l_nce + a.lambda_pull * l_pull
            n_pull_steps += 1
        else:
            l_pull = torch.zeros((), device=Vd.device)
            loss = l_nce

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        if sched is not None:
            sched.step()
        opt.zero_grad(set_to_none=True)
        fl, fn, fp = (float(loss.detach()), float(l_nce.detach()),
                      float(l_pull.detach()))
        hist.append({"step": step, "loss": fl, "nce": fn, "pull": fp,
                     "n_docs": len(doc_ids)})
        if step % 50 == 0 or step == a.steps - 1:
            log(f"  step {step:4d}/{a.steps}  loss {fl:.4f} "
                f"(nce {fn:.4f} pull {fp:.4f})  "
                f"{len(doc_ids)} docs  {time.time() - t0:.0f}s")
        if mon is not None and (step + 1) % a.monitor_every == 0:
            m = monitor_stats(mdl, tok, spec, *mon, a.monitor_bs, max_len, a.precision)
            m["step"] = step
            monlog.append(m)
            d = m["median_mutual_rank"] / monlog[0]["median_mutual_rank"] - 1
            log(f"  [monitor] step {step:4d}  "
                f"mutual rank {m['median_mutual_rank']:.2f} ({d:+.1%})  "
                f"within degree {m['pair_within_degree']:.3f}  "
                f"target overlap {m['target_overlap']:.3f}")

    out.parent.mkdir(parents=True, exist_ok=True)
    mdl.save_pretrained(str(out))
    write_json(out.parent / f"trainlog_{a.tag}.json", {
        "stage": "train", "method": "SYN", "corpus": a.corpus, "tag": a.tag,
        "encoder": a.encoder, "steps": a.steps, "batch_q": a.batch_q,
        "n_neg_per_anchor": a.n_neg, "K": int(K), "tau": a.tau,
        "lambda_pull": a.lambda_pull, "pull_tau": a.pull_tau,
        "pull_margin": a.pull_margin, "pull_grad": a.pull_grad,
        "pull_name": a.pull_name, "L": int(P["L"]), "L_lo": int(P["L_lo"]),
        "pos_select": a.pos_select, "neg_source": a.neg_source,
        "lr": a.lr, "schedule": a.schedule,
        "lora_r": a.lora_r, "lora_targets": targets, "precision": a.precision,
        "grad_checkpoint": not a.no_grad_checkpoint, "seed": a.seed,
        "n_train_queries": int(nq), "trainable_params": int(ntr),
        "negatives_per_query_mean": float(np.mean(n_neg_seen)),
        "steps_with_active_pairs": n_pull_steps,
        "elapsed_s": time.time() - t0, "history": hist, "monitor": monlog,
        **provenance(**spec_record(spec, max_len))})
    log(f"saved {out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
