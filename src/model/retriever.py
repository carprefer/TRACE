"""BGE-M3 dense retriever used by the QCRC residual-evidence fallback (§4.2.v).

Pinned to fp32 (P40s do fp16 matmul ~60x slower than fp32 — keep weights fp32).
Per-qid (hybridqa) or per-domain (sparta) index saved as .npz.
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import config

# Pin GPU before importing torch
os.environ.setdefault("CUDA_VISIBLE_DEVICES", config.RETRIEVER_GPU)

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_lock = threading.Lock()
_TOK = None
_MODEL = None


def get_encoder():
    global _TOK, _MODEL
    if _TOK is not None and _MODEL is not None:
        return _TOK, _MODEL
    with _lock:
        if _TOK is None:
            _TOK = AutoTokenizer.from_pretrained(str(config.BGE_MODEL_PATH), use_fast=True)
            _MODEL = AutoModel.from_pretrained(str(config.BGE_MODEL_PATH)).to(_DEVICE).eval()
    return _TOK, _MODEL


def _encode(texts: list, max_len: int) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1024), dtype=np.float32)
    tok, model = get_encoder()
    out = []
    bs = config.RETRIEVER_BATCH
    with torch.no_grad():
        for s in range(0, len(texts), bs):
            batch = texts[s:s + bs]
            inp = tok(batch, padding=True, truncation=True,
                      max_length=max_len, return_tensors="pt").to(_DEVICE)
            emb = model(**inp)[0][:, 0]
            emb = torch.nn.functional.normalize(emb, dim=-1)
            out.append(emb.cpu().numpy().astype(np.float32))
    return np.concatenate(out, axis=0)


def build_index(name: str, passages: dict) -> dict:
    urls = list(passages.keys())
    texts = [passages[u] or "" for u in urls]
    emb = _encode(texts, config.RETRIEVER_DOC_LEN)
    return {"urls": urls, "texts": texts, "emb": emb}


def save_index(name: str, index: dict, out_dir: Path = None) -> Path:
    out_dir = out_dir or config.SEARCH_INDEX_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{name}.npz"
    np.savez(p,
             urls=np.array(index["urls"], dtype=object),
             texts=np.array(index["texts"], dtype=object),
             emb=index["emb"])
    return p


def load_index(name: str, in_dir: Path = None) -> dict | None:
    in_dir = in_dir or config.SEARCH_INDEX_DIR
    p = in_dir / f"{name}.npz"
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    return {"urls": list(d["urls"]), "texts": list(d["texts"]), "emb": d["emb"]}


def search(index: dict, query: str, k: int = 3) -> list:
    if index is None or not index.get("urls"):
        return []
    q_emb = _encode([query], config.RETRIEVER_QRY_LEN)
    scores = (index["emb"] @ q_emb.T).flatten()
    k = min(k, len(scores))
    if k < len(scores):
        topk = np.argpartition(-scores, k - 1)[:k]
    else:
        topk = np.arange(len(scores))
    topk = topk[np.argsort(-scores[topk])]
    return [
        {"url": index["urls"][i], "text": index["texts"][i], "score": float(scores[i])}
        for i in topk
    ]
