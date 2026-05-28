"""QCRC step (iv) + (v): SQL execution agent + residual-evidence fallback (Fig. 11).

ReAct loop over D_q (the question-completed database):

Each turn the LLM emits ONE of:
  {"thought": "...", "sql": "<SELECT...>"}                -- explore
  {"thought": "...", "answer": "<short literal>"}         -- terminate
  {"thought": "...", "answer_sql": "<SELECT...>"}         -- terminate, result rows ARE the answer

Tool-set picking rule (this codebase's lock):
  * If the previous SQL observation had n_rows > MAX_ROWS_RETURNED (i.e. it was
    truncated in the prompt), the next turn's system prompt drops `answer` and
    offers `answer_sql` instead -- the model can't reliably type out the answer
    from a window that hid most of the rows.
  * Otherwise (no truncation), the model gets the default {sql, answer} set.

Residual-evidence rule (paper §4.2.v) -- this codebase's lock:
  * After each `sql` turn we always auto-attach passages linked from any
    returned cell (the value->url map built in SGR/assembly).
  * **Dense retrieval over the passage corpus runs ONLY when the SQL returned
    zero rows**, appending the top-K passages to the collected buffer. This is
    the residual-evidence fallback in the paper.
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src import config
from src.model import compact_schema, corpus, profiler, retriever as rt
from src.model.llm_client import make_client
from src.prompt import qcrc as prompt

DB_DIR = config.DB_COMPLETED

SYSTEM_PROMPT = prompt.agent_system_prompt
SYSTEM_PROMPT_TRUNC = prompt.agent_system_prompt_truncated
USER_PROMPT = prompt.agent_user_prompt


def _has_truncated_sql(transcript) -> bool:
    """True iff the MOST RECENT sql turn returned more rows than the cap.

    Note: only the *latest* SQL is checked -- an earlier truncated turn does
    not keep the trigger latched. Once the model returns to a small result set
    we revert to the default `sql + answer` tool set.
    """
    last_sql = next((s for s in reversed(transcript) if s.get("action") == "sql"), None)
    return last_sql is not None and (last_sql.get("n_rows") or 0) > config.MAX_ROWS_RETURNED


def _build_messages(schema: str, question: str, transcript, collected, *, mode: str = "next"):
    history = _format_history(transcript) or "(no prior turns)"
    has_trunc = _has_truncated_sql(transcript)
    sys_text = SYSTEM_PROMPT_TRUNC if has_trunc else SYSTEM_PROMPT
    if mode == "forced":
        last_action = "answer_sql" if has_trunc else "answer"
        history = (f"{history}\n\n(This is the final turn -- output `{last_action}` only, "
                   "no further `sql`.)")
    sys_content = sys_text.format()
    user_content = USER_PROMPT.format(question=question, schema=schema, history=history,
                                      collected=_format_collected(collected))
    return [{"role": "system", "content": sys_content.strip()},
            {"role": "user",   "content": user_content.strip()}]


def _profile_conn(conn, tables):
    profiles = {}
    for t in tables:
        df = pd.read_sql_query(f'SELECT * FROM "{t}"', conn)
        for col in df.columns:
            if df[col].dtype == "float64":
                nn = df[col].dropna()
                if not nn.empty and (nn == nn.astype(int)).all():
                    df[col] = df[col].astype("Int64")
        profiles[t] = profiler.profile_table(df, top_k=config.TOP_K_IN_PROFILE)
    return profiles


def _build_schema(conn, joinmap: dict = None) -> str:
    side_set = set((joinmap or {}).keys())
    all_tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()]
    source_tables = [t for t in all_tables if t not in side_set]
    side_tables = [t for t in all_tables if t in side_set]
    ordered = source_tables + side_tables
    row_counts = {t: conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0] for t in ordered}
    profiles = _profile_conn(conn, ordered)
    base = compact_schema.build_compact_schema(conn, profiles, ordered)

    blocks: dict = {}
    cur = None; buf: list = []
    for line in base.split("\n"):
        m = re.match(r'CREATE TABLE "(\w+)"', line)
        if m:
            if cur is not None:
                blocks[cur] = buf
            cur = m.group(1); buf = [line]
        else:
            buf.append(line)
    if cur is not None:
        blocks[cur] = buf

    join_refs = {tbl: refs for tbl, refs in (joinmap or {}).items() if refs}

    out: list = []
    if source_tables:
        out.append("-- Source tables (original structured data):")
        for t in source_tables:
            if t not in blocks:
                continue
            out.append(f'-- table "{t}": {row_counts[t]} rows')
            out.extend(blocks[t])
            out.append("")
    if side_tables:
        out.append("-- Side tables (LLM-extracted from passages -- values may be incomplete or imprecise):")
        for t in side_tables:
            if t not in blocks:
                continue
            out.append(f'-- table "{t}": {row_counts[t]} rows')
            if t in join_refs:
                out.append(f'-- linked_cell joinable with: {", ".join(join_refs[t])}')
            out.extend(blocks[t])
            out.append("")
    return "\n".join(out).rstrip()


_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


def _parse_fenced_json(raw: str):
    m = _FENCED_JSON_RE.search(raw)
    if m:
        return json.loads(m.group(1))
    return json.loads(raw.strip())


def _build_value_to_url(conn, db_path: Path = None) -> dict:
    """value -> {urls} from {db_stem}.urlmap.json. Sidecar: {side_tbl: [url_row0, ...]}."""
    m: dict = {}
    if db_path is None:
        return m
    sidecar = db_path.parent / f"{db_path.stem}.urlmap.json"
    if not sidecar.exists():
        return m
    data = json.loads(sidecar.read_text())
    for tbl, urls in data.items():
        if not isinstance(urls, list):
            continue
        try:
            rows = list(conn.execute(f'SELECT * FROM "{tbl}"'))
        except sqlite3.Error:
            continue
        for ri, row in enumerate(rows):
            if ri >= len(urls):
                break
            u = urls[ri]
            if not u:
                continue
            for v in row:
                if v is None:
                    continue
                s = str(v).strip()
                if not s:
                    continue
                m.setdefault(s, set()).add(u)
    return m


def _execute_sql(conn, sql):
    try:
        cur = conn.execute(sql); rows = cur.fetchall()
    except sqlite3.Error as e:
        return f"SQL_ERROR: {e}", [], []
    cols = [d[0] for d in cur.description] if cur.description else []
    if not rows:
        return "(0 rows)", cols, []
    body = [" | ".join(cols), "-+-".join("-" * max(len(c), 3) for c in cols)]
    for r in rows[:config.MAX_ROWS_RETURNED]:
        body.append(" | ".join("NULL" if v is None else str(v) for v in r))
    if len(rows) > config.MAX_ROWS_RETURNED:
        body.append(f"... ({len(rows)} rows total, {config.MAX_ROWS_RETURNED} shown)")
    return "\n".join(body), cols, rows


def _attach(collected, seen_urls, url, text, turn: int):
    if url in seen_urls:
        for c in collected:
            if c["url"] == url:
                if turn not in c["turns"]:
                    c["turns"].append(turn)
                return
    seen_urls.add(url)
    collected.append({"url": url, "text": text, "turns": [turn]})


def _update_collected(rows, value_to_url, passages, collected, seen_urls, turn: int):
    """Auto-attach: every cell value in returned rows that matches a side-table cell
    gets its source passage attached to `collected`."""
    for r in rows[:config.MAX_ROWS_RETURNED]:
        for v in r:
            if v is None:
                continue
            for u in value_to_url.get(str(v), ()):
                text = passages.get(u, "(no passage)") if passages else "(no passage)"
                _attach(collected, seen_urls, u, text, turn)


def _residual_search(search_index, query: str, passages, collected, seen_urls, turn: int,
                     k: int = None):
    """Residual-evidence fallback (paper §4.2.v): runs only when triggered by an
    empty-result SQL turn -- see `run()` for the trigger."""
    if k is None:
        k = config.SEARCH_TOP_K
    if search_index is None or not query:
        return
    try:
        hits = rt.search(search_index, query, k=k)
    except Exception:
        return
    for h in hits:
        u = h.get("url")
        if not u:
            continue
        text = (passages.get(u) if passages else None) or h.get("text") or "(no passage)"
        _attach(collected, seen_urls, u, text, turn)


def _format_collected(entries) -> str:
    if not entries:
        return "(none yet)"
    ranked = sorted(
        entries,
        key=lambda c: (-len(c.get("turns") or []), -max(c.get("turns") or [0])),
    )[:config.MAX_COLLECTED_SHOWN]
    parts = []
    if len(entries) > config.MAX_COLLECTED_SHOWN:
        parts.append(f"(showing top {config.MAX_COLLECTED_SHOWN} of {len(entries)} "
                     f"collected, ranked by turn-coverage then recency)")
    for i, c in enumerate(ranked, 1):
        turns_str = ", ".join(str(t) for t in c["turns"])
        parts.append(f"[{i}] (collected on turn {turns_str})")
        parts.append(c["text"])
    return "\n".join(parts)


def _format_history(transcript) -> str:
    if not transcript:
        return ""
    parts = []
    for s in transcript:
        parts.append(f'[turn {s["turn"]}]')
        if s.get("thought"):
            parts.append(f'  thought: {s["thought"]}')
        a = s.get("action")
        if a == "sql":
            parts.append(f'  sql: {s.get("input","")}')
            if s.get("observation") is not None:
                parts.append(f'  observation:')
                for line in str(s["observation"]).split("\n"):
                    parts.append(f"    {line}")
        elif a == "answer":
            parts.append(f'  answer: {s.get("input","")}')
        else:
            parts.append(f'  ({a or "no-action"}: {s.get("input","")})')
    return "\n".join(parts)


def _execute_answer_sql(conn, sql):
    """Run an answer_sql query and reduce to a flat pred list (first column, capped)."""
    if not sql or not isinstance(sql, str):
        return None, None, "no_sql"
    try:
        cur = conn.execute(sql); rows = cur.fetchall()
    except sqlite3.Error as e:
        return None, None, str(e)
    n = len(rows)
    vals = [str(r[0]) for r in rows[:config.ANSWER_SQL_ROW_CAP] if r and r[0] is not None]
    if not vals:
        return None, n, None
    pred = vals if len(vals) > 1 else vals[0]
    return pred, n, None


def _call_llm(client, messages):
    t0 = time.perf_counter()
    raw = client.inference(messages=messages, max_new_tokens=2048, temperature=0.0)
    dt_ms = (time.perf_counter() - t0) * 1000
    u = client.get_last_call_usage() or {}
    usage = {"prompt_tokens": u.get("prompt_tokens", 0),
             "completion_tokens": u.get("completion_tokens", 0)}
    return raw, usage, dt_ms


def _resolve_run_inputs(qid: str):
    """Return (rec, db_path, passages, search_index_name)."""
    if config.BENCHMARK == "hybridqa":
        rec = corpus.get_record(qid)
        return rec, DB_DIR / f"{qid}.db", (rec.get("text") or {}), qid
    if config.BENCHMARK == "sparta":
        rec = corpus.get_question(qid)
        qid_db = DB_DIR / f"{qid}.db"
        domain_db = DB_DIR / f"{config.DOMAIN}.db"
        db = qid_db if qid_db.exists() else domain_db
        return rec, db, corpus.get_text_data(), "all"
    raise NotImplementedError(f"benchmark {config.BENCHMARK!r}")


def run(qid: str, client=None) -> dict:
    if client is None:
        client = make_client()
    rec, db_path, passages, idx_name = _resolve_run_inputs(qid)
    if not db_path.exists():
        return {"qid": qid, "pred": None, "gold": rec.get("answer"), "error": "no_db"}
    conn = sqlite3.connect(db_path)
    joinmap_path = db_path.parent / f"{db_path.stem}.joinmap.json"
    joinmap = json.loads(joinmap_path.read_text()) if joinmap_path.exists() else {}
    schema = _build_schema(conn, joinmap=joinmap)
    value_to_url = _build_value_to_url(conn, db_path)
    search_index = rt.load_index(idx_name)

    transcript = []
    final = None
    totals = {"prompt": 0, "completion": 0, "llm_ms": 0.0, "calls": 0}
    question = rec.get("question", "")

    collected: list = []
    seen_urls: set = set()

    for step in range(1, config.MAX_STEPS + 1):
        messages = _build_messages(schema, question, transcript, collected, mode="next")
        raw, usage, dt_ms = _call_llm(client, messages)
        totals["prompt"]     += usage["prompt_tokens"]
        totals["completion"] += usage["completion_tokens"]
        totals["llm_ms"]     += dt_ms; totals["calls"] += 1

        try:
            obj = _parse_fenced_json(raw)
        except Exception:
            obj = {"thought": "(parse error)", "answer": raw}
        thought = obj.get("thought", "")
        if "answer_sql" in obj:
            action = "answer_sql"; action_input = obj.get("answer_sql", "")
        elif "answer" in obj:
            action = "answer"; action_input = obj.get("answer", "")
        elif "sql" in obj:
            action = "sql"; action_input = obj.get("sql", "")
        else:
            action = ""; action_input = ""
        transcript.append({"turn": step, "thought": thought, "action": action,
                           "input": action_input})

        if action == "answer":
            final = action_input; break
        if action == "answer_sql":
            pred, n_rows, err = _execute_answer_sql(conn, action_input)
            transcript[-1]["answer_sql_n_rows"] = n_rows
            if err:
                transcript[-1]["answer_sql_err"] = err
            final = pred; break
        if action == "sql":
            obs, _, rows = _execute_sql(conn, action_input)
            transcript[-1]["n_rows"] = len(rows)
            _update_collected(rows, value_to_url, passages, collected, seen_urls, turn=step)
            # *** Residual-evidence fallback: only when SQL returns 0 rows. ***
            if len(rows) == 0:
                _residual_search(search_index, thought, passages, collected, seen_urls, turn=step)
        else:
            obs = f"UNKNOWN_ACTION: {action}"
        transcript[-1]["observation"] = obs

    # MAX_STEPS exhausted -> force one final answer call.
    if final is None:
        messages = _build_messages(schema, question, transcript, collected, mode="forced")
        raw, usage, dt_ms = _call_llm(client, messages)
        totals["prompt"]     += usage["prompt_tokens"]
        totals["completion"] += usage["completion_tokens"]
        totals["llm_ms"]     += dt_ms; totals["calls"] += 1
        forced_obj = None; forced_input = None; forced_action = "answer"
        forced_meta: dict = {}
        try:
            forced_obj = _parse_fenced_json(raw)
            if isinstance(forced_obj, dict):
                if "answer_sql" in forced_obj:
                    forced_action = "answer_sql"
                    pred, n_rows, err = _execute_answer_sql(conn, forced_obj.get("answer_sql"))
                    forced_input = pred
                    forced_meta = {"answer_sql_n_rows": n_rows,
                                   **({"answer_sql_err": err} if err else {})}
                else:
                    forced_input = forced_obj.get("answer")
        except Exception:
            pass
        final = forced_input if forced_input is not None else raw
        transcript.append({
            "turn": "forced_answer",
            "thought": (forced_obj.get("thought") if isinstance(forced_obj, dict) else ""),
            "action": forced_action,
            "input": final,
            "forced": True,
            **forced_meta,
        })

    conn.close()
    return {"qid": qid, "pred": final, "gold": rec.get("answer"),
            "transcript": transcript, "totals": totals}
