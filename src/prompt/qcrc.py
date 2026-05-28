"""Prompts used by Question-Conditioned Relational Completion (QCRC; paper §4.2).

Indexed against the appendix table (Fig. 9 -> 11). Reproduced verbatim from
the implementation. Placeholders use {curly_braces}.
"""

# -----------------------------------------------------------------------------
# Fig. 9 -- Question-conditioned planning (Step i of QCRC)
# -----------------------------------------------------------------------------

planning_system_prompt = """You are a planner for a database QA task.
You are given a natural-language question and a SQLite DB schema. The DB has two kinds of tables:
- Source tables: original structured data.
- Side tables: derived by LLM extraction from passages — values may be incomplete or imprecise.

Your job: decide which tables and columns are required to answer the question.
If the existing columns are insufficient, you may propose new attributes to be
extracted from passages — but new attributes can only be added to side tables
(since they are extracted from passages). Source tables cannot gain new columns.

Output STRICT JSON:
```json
{{
  "needed": [
    {{
      "table": "<table>",
      "columns": ["<existing_col1>", "<existing_col2>", ...],
      "new_columns": [
        {{"name": "<snake_case>", "type": "<int|float|date|text>", "description": "<ONE short sentence describing what this attribute represents>"}},
        ...
      ]
    }},
    ...
  ]
}}
```
- Use exact table/column names from the schema for existing columns.
- Omit `new_columns` (or use an empty list) if the existing columns are sufficient.
- `new_columns` may appear ONLY on side tables (not source tables).
- Propose `new_columns` only when no existing column already captures the needed information.
- Each `description` MUST be a single short sentence (no enumeration, no multi-step reasoning)."""


planning_user_prompt = """Question: {question}

DB SCHEMA:
/*
{schema}
*/

Based on the question and the schema, list the tables and columns required to answer."""


# -----------------------------------------------------------------------------
# Fig. 10 -- Targeted extraction (Step iii of QCRC)
# -----------------------------------------------------------------------------

completion_system_prompt = """You extract one attribute value for a subject from a passage.
The original question is provided as context — extract the value that is most relevant for answering it.
Only extract what the passage states about the subject. Use null if the passage does not state the attribute.

Type rules for the output value:
- int   -> plain integer (no thousand separators, no units).
- float -> plain number (no thousand separators, no units).
- date  -> 'YYYY-MM-DD'. Year-only -> 'YYYY-01-01'; month/day-only -> '9999-MM-DD'.
- text  -> short literal string copied from the passage.

Output STRICT JSON: {{"value": <typed value> | null}}."""


completion_user_prompt = """original_question (context): {question}

attribute: {name} ({type}) -- {description}

subject (cell_value): {cell_value}

passage:
{passage}

Extract the attribute value for this subject."""


# -----------------------------------------------------------------------------
# Fig. 11 -- SQL execution over the completed state (Step iv of QCRC)
# -----------------------------------------------------------------------------
#
# Two variants — the system prompt swap is controlled by whether the most
# recent SQL turn produced more rows than the cap (n_rows > MAX_ROWS_RETURNED):
#
#   * truncated == False -> {sql, answer}      (this file: agent_system_prompt)
#   * truncated == True  -> {sql, answer_sql}  (this file: agent_system_prompt_truncated)
#
# Rationale: when the prior SELECT clearly returned more rows than the
# observation window can show, the model can't reliably type out the answer
# from memory — so we DROP the bare-string `answer` mode and offer only
# `answer_sql`, whose result rows ARE the answer (capped at ANSWER_SQL_ROW_CAP).

agent_system_prompt = """You are ReAct style QA expert.
You are working on SQLite DB.

Each turn, you can execute sql to find more information.
If enough information is gathered, you must answer the original question.

Output strict JSON:
```json
{{
  "thought": "...",
  "sql": "<a self-contained SQLite SELECT>"
}}
```
or
```json
{{
  "thought": "...",
  "answer": "<short literal value(s)>"
}}
```"""


agent_system_prompt_truncated = """You are ReAct style QA expert.
You are working on SQLite DB.

Each turn, you can execute sql to find more information.
If enough information is gathered, you must answer the original question.

Output strict JSON:
```json
{{
  "thought": "...",
  "sql": "<a self-contained SQLite SELECT>"
}}
```
or
```json
{{
  "thought": "...",
  "answer_sql": "<a self-contained SQLite SELECT — its output rows ARE the answer>"
}}
```"""


agent_user_prompt = """
Original Question: {question}

DB SCHEMA:
/*
{schema}
*/


History:
{history}

Collected passages (cumulative, in attach order):
{collected}

Based on above information, select right action."""
