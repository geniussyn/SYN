"""Encoder registry: pooling and prefixes follow each model card.

Getting either wrong makes an encoder look unfairly bad and inflates its
fragmentation, so the applied strings and the measured truncation rate are
recorded in every output manifest.
"""
import contextlib
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from syn.env import log  # noqa: E402


@dataclass(frozen=True)
class EncoderSpec:
    hf_id: str
    dim: int                       # output dimension, after MRL truncation if any
    pooling: str                   # "cls" | "mean" | "last"
    query_prefix: str = ""
    doc_prefix: str = ""
    max_len: int = 512
    revision: str | None = None
    trust_remote_code: bool = False
    mrl: bool = False              # truncate native dim to `dim`, then renormalise
    load_dtype: str | None = None
    autocast: bool = True          # False: native precision, no bf16 autocast
    notes: str = ""


REGISTRY: dict[str, EncoderSpec] = {
    "bge-m3": EncoderSpec(
        "BAAI/bge-m3", 1024, "cls",
        revision="5617a9f61b028005a4858fdac845db406aefb181",
        notes="Reference encoder. MS MARCO cells use max_len=256."),
    "bge-large-en-v1.5": EncoderSpec(
        "BAAI/bge-large-en-v1.5", 1024, "cls",
        query_prefix="Represent this sentence for searching relevant passages: "),
    "bge-small-en-v1.5": EncoderSpec(
        "BAAI/bge-small-en-v1.5", 384, "cls",
        query_prefix="Represent this sentence for searching relevant passages: "),
    "all-MiniLM-L6-v2": EncoderSpec(
        "sentence-transformers/all-MiniLM-L6-v2", 384, "mean",
        notes="Mean pooling; CLS breaks this checkpoint."),
    "e5-base-v2": EncoderSpec(
        "intfloat/e5-base-v2", 768, "mean",
        query_prefix="query: ", doc_prefix="passage: "),
    "e5-large-v2": EncoderSpec(
        "intfloat/e5-large-v2", 1024, "mean",
        query_prefix="query: ", doc_prefix="passage: ",
        notes="24 layers x d=1024, so LoRA capacity matches bge-m3 exactly."),
    # Alibaba-NLP/gte-large-en-v1.5 ships its own modelling code, and that code
    # indexes incorrectly under transformers 5.x (IndexError inside its own
    # embedding path, or a device-side assert on GPU). thenlper/gte-large is the
    # same family at the same width on a stock BERT backbone, so the encoder
    # ladder keeps its GTE rung without pinning an old transformers.
    "gte-large": EncoderSpec(
        "thenlper/gte-large", 1024, "mean",
        notes="Stock BERT backbone; the v1.5 checkpoint's remote code is "
              "incompatible with transformers 5.x."),
    "gte-large-en-v1.5": EncoderSpec(
        "Alibaba-NLP/gte-large-en-v1.5", 1024, "cls", trust_remote_code=True),
    "modernbert-embed-large": EncoderSpec(
        "lightonai/modernbert-embed-large", 1024, "mean",
        query_prefix="search_query: ", doc_prefix="search_document: "),
    "qwen3-embedding-0.6b": EncoderSpec(
        "Qwen/Qwen3-Embedding-0.6B", 1024, "last",
        query_prefix="Instruct: Given a web search query, retrieve relevant passages "
                     "that answer the query\nQuery: "),
    "qwen3-embedding-4b": EncoderSpec(
        "Qwen/Qwen3-Embedding-4B", 1024, "last", mrl=True,
        query_prefix="Instruct: Given a web search query, retrieve relevant passages "
                     "that answer the query\nQuery: ",
        notes="Native 2560d truncated to 1024d so the size ladder is dimension-matched."),
    # The two ultra-high-dimensional arms. The 4B entry above is MRL-truncated
    # to 1024 so a size ladder could hold dimension fixed; these are the native
    # widths, which is the opposite question: does the repair still have work to
    # do when the space is wide enough that the encoder could keep the top-10
    # apart on its own.
    "qwen3-embedding-4b-native": EncoderSpec(
        "Qwen/Qwen3-Embedding-4B", 2560, "last",
        query_prefix="Instruct: Given a web search query, retrieve relevant passages "
                     "that answer the query\nQuery: ",
        notes="Native 2560d, no MRL truncation."),
    "qwen3-embedding-8b": EncoderSpec(
        "Qwen/Qwen3-Embedding-8B", 4096, "last",
        query_prefix="Instruct: Given a web search query, retrieve relevant passages "
                     "that answer the query\nQuery: ",
        notes="Native 4096d. 1M docs at fp16 is 8 GiB per arm per collection."),
    "qzhou-embedding": EncoderSpec(
        "Kingsoft-LLM/QZhou-Embedding", 3584, "mean", trust_remote_code=True,
        load_dtype="bfloat16", autocast=False,
        query_prefix="Instruct: Given a web search query, retrieve relevant passages "
                     "that answer the query\nQuery: ",
        notes="NaN under bf16 autocast; runs fp32 with TF32 matmul. LoRA targets "
              "q_proj/k_proj/v_proj, not BERT names."),
}

# LoRA target modules by backbone naming convention.
LORA_TARGETS = {"bert": ["query", "key", "value"],
                "llama": ["q_proj", "k_proj", "v_proj"]}


def lora_targets(spec: EncoderSpec) -> list[str]:
    return LORA_TARGETS["llama" if "qwen" in spec.hf_id.lower()
                        or "qzhou" in spec.hf_id.lower() else "bert"]


def _dtype_kw(dtype):
    import transformers
    major = int(transformers.__version__.split(".")[0])
    return {"dtype": dtype} if major >= 5 else {"torch_dtype": dtype}


def _compat_shims():
    """Restore two cache APIs removed in transformers 5.x, for checkpoints that
    ship their own 4.x modelling code. Inference only."""
    import transformers
    C = getattr(transformers, "DynamicCache", None)
    if C is not None and not hasattr(C, "from_legacy_cache"):
        @classmethod
        def from_legacy_cache(cls, past_key_values=None):
            cache = cls()
            for layer, kv in enumerate(past_key_values or []):
                cache.update(kv[0], kv[1], layer)
            return cache
        C.from_legacy_cache = from_legacy_cache
    if C is not None and not hasattr(C, "get_max_length"):
        C.get_max_length = lambda self: None


def load_encoder(name: str, device="cuda", dtype=torch.float32,
                 allow_remote_code=False):
    spec = REGISTRY[name]
    if not spec.autocast:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    if spec.load_dtype and dtype is torch.float16:
        dtype = getattr(torch, spec.load_dtype)
    if spec.trust_remote_code and not allow_remote_code:
        raise SystemExit(f"{name} requires remote code; pass --allow-remote-code.")
    from transformers import AutoModel, AutoTokenizer
    kw = {"trust_remote_code": spec.trust_remote_code}
    if spec.revision:
        kw["revision"] = spec.revision
    tok = AutoTokenizer.from_pretrained(spec.hf_id, **kw)
    if spec.trust_remote_code:
        _compat_shims()
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(spec.hf_id, **kw)
        rp = getattr(cfg, "rope_parameters", None) or {}
        for key in ("rope_theta", "rope_scaling"):
            if not hasattr(cfg, key) and rp.get(key) is not None:
                setattr(cfg, key, rp[key])
        cfg.use_cache = False
        mdl = AutoModel.from_pretrained(spec.hf_id, config=cfg,
                                        **_dtype_kw(dtype), **kw).to(device)
    else:
        mdl = AutoModel.from_pretrained(spec.hf_id, **_dtype_kw(dtype), **kw).to(device)
    mdl.eval()
    return tok, mdl, spec


def _tokenize(tok, texts, max_len, spec: EncoderSpec):
    return tok(texts, padding=True, truncation=True, max_length=max_len,
               return_tensors="pt")


def _pool(out, mask, pooling):
    h = out.last_hidden_state
    if pooling == "cls":
        return h[:, 0]
    if pooling == "mean":
        m = mask.unsqueeze(-1).to(h.dtype)
        return (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
    if pooling == "last":
        idx = mask.sum(1).long() - 1
        return h[torch.arange(h.size(0), device=h.device), idx.clamp(min=0)]
    raise ValueError(pooling)


def _project(v, spec: EncoderSpec):
    if spec.mrl:
        if v.shape[-1] < spec.dim:
            raise SystemExit(f"{spec.hf_id}: native {v.shape[-1]}d < target {spec.dim}d")
        v = v[:, :spec.dim]
    elif v.shape[-1] != spec.dim:
        raise SystemExit(f"{spec.hf_id}: output {v.shape[-1]}d != registered {spec.dim}d")
    return torch.nn.functional.normalize(v.float(), dim=-1)


def forward_embed(mdl, tok, spec: EncoderSpec, texts, is_query, device="cuda",
                  max_len=None, precision="fp32"):
    """Differentiable encode. `precision` selects the autocast dtype of the
    forward pass; the pooled vector is normalised in fp32 either way."""
    max_len = max_len or spec.max_len
    prefix = spec.query_prefix if is_query else spec.doc_prefix
    if prefix:
        texts = [prefix + t for t in texts]
    enc = _tokenize(tok, texts, max_len, spec).to(device)
    ctx = (torch.autocast("cuda", dtype=torch.bfloat16) if precision == "bf16"
           else contextlib.nullcontext())
    with ctx:
        v = _pool(mdl(**enc), enc["attention_mask"], spec.pooling)
    return _project(v, spec)


@torch.no_grad()
def encode(mdl, tok, spec: EncoderSpec, texts, is_query: bool, device="cuda",
           bs=256, max_len=None, log_every=100, sort_by_length=True,
           precision=None):
    """(n, dim) float32 unit-normalised vectors, and the measured truncation rate.

    `precision=None` follows the spec, which is what the served index was built
    with; pass "fp32" or "bf16" to override. Batches are length-sorted to remove
    padding waste -- the attention mask makes this output-identical -- and the
    original order is restored before returning.
    """
    max_len = max_len or spec.max_len
    prefix = spec.query_prefix if is_query else spec.doc_prefix
    n = len(texts)
    order = (np.argsort([len(t) for t in texts], kind="stable")
             if sort_by_length else np.arange(n))
    out = np.empty((n, spec.dim), dtype=np.float32)
    autocast = spec.autocast if precision is None else (precision == "bf16")
    n_trunc = 0
    for bi, s in enumerate(range(0, n, bs)):
        idx = order[s:min(s + bs, n)]
        chunk = [prefix + texts[i] for i in idx] if prefix else [texts[i] for i in idx]
        enc = _tokenize(tok, chunk, max_len, spec).to(device)
        # Counted on the truncated length: a document at exactly max_len is
        # counted as truncated, which over-reports slightly and never under-reports.
        n_trunc += int((enc["attention_mask"].sum(1) >= max_len).sum())
        ctx = (torch.autocast("cuda", dtype=torch.bfloat16) if autocast
               else contextlib.nullcontext())
        with ctx:
            v = _pool(mdl(**enc), enc["attention_mask"], spec.pooling)
        out[idx] = _project(v, spec).cpu().numpy()
        if log_every and bi % log_every == 0:
            log(f"    encoded {min(s + bs, n):,}/{n:,}")
    rate = n_trunc / max(n, 1)
    log(f"  truncated {n_trunc:,}/{n:,} ({rate:.2%}) - max_len={max_len} "
        f"pooling={spec.pooling} prefix={prefix!r} autocast={autocast}")
    return out, rate


def spec_record(spec: EncoderSpec, max_len=None) -> dict:
    return {"encoder_hf_id": spec.hf_id, "encoder_revision": spec.revision,
            "dim": spec.dim, "pooling": spec.pooling,
            "query_prefix": spec.query_prefix, "doc_prefix": spec.doc_prefix,
            "max_len": max_len or spec.max_len,
            "trust_remote_code": spec.trust_remote_code,
            "mrl_truncated": spec.mrl}
