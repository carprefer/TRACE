"""Post-hoc evaluator — reads agent_*.summary.jsonl produced by 11_agent.py and
computes the four metrics reported in the paper:

  * EM, F1, P, R  -- token-set comparison (matches TextDBQA/analysis/evaluation.py
                     `compute_prfem_token`, the function used to produce the
                     paper's Table 1 numbers). pred and gold lists are joined
                     with a space, .split()-tokenized, and compared as sets.
  * LJ            -- LLM-as-Judge using gpt-4o (paper §A.2, Fig. 12).
                     OPT-IN: only runs when --judge is passed (costs OpenAI tokens).

Notes:
  * LJ requires OPENAI_API_KEY (regardless of TRACE_LLM).

Usage:
  python -u 12_eval.py [--summary <path>] [--judge] [--workers 16]

If --summary is not given, picks the latest agent_*.summary.jsonl under
config.RESULTS_DIR.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src import config
from src.model import corpus
from src.prompt import eval as eval_prompt


# ---------- metrics (ported verbatim from TextDBQA/analysis/evaluation.py) ----

def _as_list(v) -> list:
    """Coerce summary.jsonl `pred` / `gold` field into a list[str]."""
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v]
    return [str(v)]


def compute_prfem_token(predict_list, answer_list) -> dict:
    """Verbatim port of analysis/evaluation.py:compute_prfem_token.

    Joins both lists with spaces, whitespace-tokenizes, converts to sets, and
    reports set-overlap P/R/F1 + set-equality EM. This is what produced the
    paper's Table 1 EM/F1/P/R columns.
    """
    predict_str = " ".join(predict_list)
    answer_str  = " ".join(answer_list)
    predict_tokens = predict_str.strip().split()
    answer_tokens  = answer_str.strip().split()
    predict_set = set(predict_tokens)
    answer_set  = set(answer_tokens)

    true_positive = len(predict_set & answer_set)
    predicted = len(predict_set)
    actual    = len(answer_set)

    precision = true_positive / predicted if predicted > 0 else 0.0
    recall    = true_positive / actual    if actual    > 0 else 0.0
    f1 = ((2 * precision * recall / (precision + recall))
          if (precision + recall) > 0 else 0.0)
    em = 1 if predict_set == answer_set else 0
    return {"precision": precision, "recall": recall, "f1": f1, "em": em}


# ---------- LLM-as-Judge -----------------------------------------------------

_VERDICT_RE = re.compile(r"\[\[(CORRECT|INCORRECT)\]\]")


def _to_json_list(v):
    if v is None:
        return json.dumps([])
    if isinstance(v, list):
        return json.dumps([str(x) for x in v], ensure_ascii=False)
    return json.dumps([str(v)], ensure_ascii=False)


def judge_one(client, question: str, gold, pred) -> dict:
    """Single judge call. Returns {'verdict': 'CORRECT'|'INCORRECT'|None, 'raw': str}."""
    msgs = [
        {"role": "system", "content": eval_prompt.judge_system_prompt},
        {"role": "user",   "content": eval_prompt.judge_user_prompt.format(
            question=question, gold_list=_to_json_list(gold), prediction_list=_to_json_list(pred))},
    ]
    raw = ""
    for _attempt in range(2):  # one retry on parse fail per paper
        try:
            raw = client.inference(messages=msgs, max_new_tokens=256, temperature=0.0)
        except Exception as e:
            return {"verdict": None, "raw": f"ERROR: {e}"}
        m = _VERDICT_RE.search(raw or "")
        if m:
            return {"verdict": m.group(1), "raw": raw}
    return {"verdict": None, "raw": raw}


def _make_judge_client():
    from omegaconf import OmegaConf
    from src.llm import load_llm
    cfg = OmegaConf.create({
        "name": "gpt-4o-mini",
        "max_tokens": 120_000,
        "max_new_tokens": 256,
        "temperature": 0,
    })
    # Paper uses gpt-4o (not -mini). Allow override via TRACE_JUDGE_MODEL.
    import os
    cfg.name = os.environ.get("TRACE_JUDGE_MODEL", "gpt-4o")
    return load_llm(cfg)


# ---------- main -------------------------------------------------------------

def _latest_summary():
    candidates = sorted(config.RESULTS_DIR.glob("agent_*.summary.jsonl"))
    if not candidates:
        return None
    return candidates[-1]


def _question_for_qid(qid: str) -> str:
    if config.BENCHMARK == "hybridqa":
        return corpus.get_record(qid).get("question", "")
    return corpus.get_question(qid).get("question", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="",
                    help="Path to agent_*.summary.jsonl. Default: latest under RESULTS_DIR.")
    ap.add_argument("--judge", action="store_true",
                    help="Run LLM-as-Judge (gpt-4o by default; needs OPENAI_API_KEY). "
                         "Default off -- only EM/F1/P/R are computed without this flag.")
    ap.add_argument("--workers", type=int, default=16,
                    help="Judge concurrency (OpenAI calls).")
    ap.add_argument("--out", default="",
                    help="Optional path to write per-qid eval JSONL "
                         "(default: <summary>.eval.jsonl alongside the summary).")
    args = ap.parse_args()

    corpus.preload()
    summary_path = Path(args.summary) if args.summary else _latest_summary()
    if summary_path is None or not summary_path.exists():
        raise SystemExit(f"no summary found (looked under {config.RESULTS_DIR})")
    out_path = Path(args.out) if args.out else summary_path.with_suffix(".eval.jsonl")
    print(f"## summary: {summary_path}")
    print(f"## eval out: {out_path}")

    rows = [json.loads(line) for line in summary_path.open() if line.strip()]
    print(f"## n_rows: {len(rows)}")

    # token-set metrics (compute_prfem_token, verbatim from analysis/evaluation.py)
    for r in rows:
        res = compute_prfem_token(_as_list(r.get("pred")), _as_list(r.get("gold")))
        r["em"]        = res["em"]
        r["precision"] = res["precision"]
        r["recall"]    = res["recall"]
        r["f1"]        = res["f1"]
        r["question"]  = _question_for_qid(r["qid"])

    if args.judge:
        import os
        if not os.environ.get("OPENAI_API_KEY"):
            print("## OPENAI_API_KEY not set -- cannot run LJ. Set the env var and retry.",
                  file=sys.stderr)
        else:
            print(f"## running LJ via {os.environ.get('TRACE_JUDGE_MODEL', 'gpt-4o')} "
                  f"on {len(rows)} rows with workers={args.workers} ...", flush=True)
            client = _make_judge_client()
            done = 0; t0 = time.time()

            def _do(row):
                v = judge_one(client, row["question"], row.get("gold"), row.get("pred"))
                return row, v

            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = [ex.submit(_do, r) for r in rows]
                for fut in as_completed(futs):
                    row, v = fut.result()
                    row["lj_verdict"] = v["verdict"]
                    row["lj_raw"] = v["raw"]
                    row["lj"] = int(v["verdict"] == "CORRECT")
                    done += 1
                    if done % 25 == 0 or done == len(rows):
                        elapsed = time.time() - t0
                        print(f"  [{done}/{len(rows)}]  elapsed={elapsed:.0f}s", flush=True)

    # write per-qid eval
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    # aggregate
    n = len(rows)
    em = 100 * sum(r["em"] for r in rows) / n
    f1 = 100 * sum(r["f1"] for r in rows) / n
    p  = 100 * sum(r["precision"] for r in rows) / n
    rr = 100 * sum(r["recall"] for r in rows) / n
    print()
    print(f"=== AGGREGATE (n={n}) ===")
    print(f"  EM:  {em:.2f}")
    print(f"  F1:  {f1:.2f}")
    print(f"  P :  {p:.2f}")
    print(f"  R :  {rr:.2f}")
    if any("lj" in r for r in rows):
        lj_valid = [r for r in rows if r.get("lj_verdict") in ("CORRECT", "INCORRECT")]
        lj_score = 100 * sum(r["lj"] for r in lj_valid) / len(lj_valid) if lj_valid else 0
        lj_err = n - len(lj_valid)
        print(f"  LJ:  {lj_score:.2f}  (judge_err={lj_err})")


if __name__ == "__main__":
    main()
