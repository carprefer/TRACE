"""SGR step (iii) runner -- attribute-value extraction (Fig. 7).

HybridQA: per-qid file extraction/<qid>.json with `tables: [{table_name, attr, rows}]`.
SPARTA  : per-table files extraction/<table_name>.json with `{table_name, attr, rows}`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src import config
from src.model import corpus
from src.model.llm_client import make_client
from src.model.sgr import grouping, schema_induction as ss, prompt_design as pd_, extraction as ext


def _llm_call(client, system, user, max_new_tokens=512):
    return client.inference(
        messages=[{"role": "system", "content": system},
                  {"role": "user",   "content": user}],
        max_new_tokens=max_new_tokens, temperature=0.0,
        response_format={"type": "json_object"},
    )


def _llm_task(t, client):
    try:
        raw = _llm_call(client, t["system"], t["user"])
    except Exception as e:
        return {**t, "raw": None, "values": {}, "parse_error": None, "error": str(e)}
    try:
        obj = json.loads(raw)
    except Exception as e:
        return {**t, "raw": raw, "values": {}, "parse_error": str(e), "error": None}
    values, parse_err = ext.coerce_values(obj)
    return {**t, "raw": raw, "values": values, "parse_error": parse_err, "error": None}


def _build_tasks_for_name(name: str):
    grp = grouping.load(name)
    sch = ss.load(name)
    spr = pd_.load(name)
    if grp is None or sch is None or spr is None:
        return []
    link_hits = ext._link_hits_from_grouping(grp)
    passages = ext._passages_from_corpus(grp)
    src = grp.get("source_table")
    spr_by_tn = {t["table_name"]: t for t in (spr.get("tables") or [])}
    tasks = []
    for t in (sch.get("tables") or []):
        attrs = t.get("attr") or []
        if not attrs:
            continue
        meta = spr_by_tn.get(t["table_name"])
        if meta is None or "overview" not in meta:
            continue
        tn = t["table_name"]
        for url, cv in ext._url_cv_pairs(link_hits, t.get("linked_columns") or [], src):
            text = passages.get(url)
            if not text:
                continue
            tasks.append({
                "name": name, "qid": grp.get("qid"), "benchmark": grp.get("benchmark"),
                "table_name": tn, "attr": attrs, "url": url, "cell_value": cv,
                "system": ext.SYSTEM_EXTRACT,
                "user":   ext.build_user(tn, attrs, meta, cv, text),
            })
    return tasks


def main_hybridqa(args):
    out_dir = config.DATA_DIR / "extraction"
    out_dir.mkdir(parents=True, exist_ok=True)
    qids = corpus.load_qids(args.qids)
    if not args.force:
        qids = [q for q in qids if not (out_dir / f"{q}.json").exists()]
    if not qids:
        print("## nothing to do."); return
    workers = args.workers or config.default_workers()
    all_tasks = []
    for q in qids:
        all_tasks.extend(_build_tasks_for_name(q))
    print(f"## qids={len(qids)}  rows={len(all_tasks)}  workers={workers}", flush=True)
    if not all_tasks:
        return
    client = make_client()
    by_qid = defaultdict(list)
    t1 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_llm_task, t, client) for t in all_tasks]
        for fut in as_completed(futs):
            r = fut.result()
            by_qid[r["qid"]].append(r)
    saved = 0
    for q in qids:
        rows = by_qid.get(q) or []
        by_table = defaultdict(list); attr_by_table = {}
        for r in rows:
            by_table[r["table_name"]].append(r)
            attr_by_table.setdefault(r["table_name"], r["attr"])
        tables_out = []
        for tn, items in by_table.items():
            tables_out.append({
                "table_name": tn, "attr": attr_by_table[tn], "n_rows": len(items),
                "rows": [{"url": it["url"], "cell_value": it.get("cell_value"),
                          "values": it["values"], "parse_error": it.get("parse_error")}
                         for it in items],
            })
        if not tables_out:
            continue
        ext.save({"qid": q, "benchmark": "hybridqa", "tables": tables_out},
                 name=q, out_dir=out_dir)
        saved += 1
    print(f"## saved {saved}  wall={time.time()-t1:.1f}s", flush=True)


def main_sparta(args):
    config.require_domain()
    out_dir = config.DATA_DIR / "extraction"
    out_dir.mkdir(parents=True, exist_ok=True)
    spr = pd_.load("all")
    if spr is None:
        print("## no prompt_design artifact; run 04_prompt_design first.")
        return
    table_names = [t["table_name"] for t in (spr.get("tables") or [])]
    if not args.force:
        existing = {fp.stem for fp in out_dir.glob("*.json")}
        target = [tn for tn in table_names if tn not in existing]
    else:
        target = list(table_names)
    if not target:
        print("## nothing to do."); return
    workers = args.workers or config.default_workers()
    all_tasks = _build_tasks_for_name("all")
    all_tasks = [t for t in all_tasks if t["table_name"] in target]
    print(f"## rows={len(all_tasks)}  tables={len(target)}  workers={workers}", flush=True)
    if not all_tasks:
        return
    client = make_client()
    by_table = defaultdict(list); attr_by_table = {}
    t1 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_llm_task, t, client) for t in all_tasks]
        for fut in as_completed(futs):
            r = fut.result()
            by_table[r["table_name"]].append(r)
            attr_by_table.setdefault(r["table_name"], r["attr"])
    for tn, items in by_table.items():
        ext.save({
            "table_name": tn, "attr": attr_by_table[tn], "n_rows": len(items),
            "rows": [{"url": it["url"], "cell_value": it.get("cell_value"),
                      "values": it["values"], "parse_error": it.get("parse_error")}
                     for it in items],
        }, name=tn, out_dir=out_dir)
    print(f"## saved {len(by_table)} tables  wall={time.time()-t1:.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qids", default="all",
                    help="'all' (default) or a JSON file path. Ignored for sparta.")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    corpus.preload()
    if config.BENCHMARK == "hybridqa":
        main_hybridqa(args)
    elif config.BENCHMARK == "sparta":
        main_sparta(args)
    else:
        raise SystemExit(config.BENCHMARK)


if __name__ == "__main__":
    main()
