"""Prompts used by Schema-Guided Relationalization (SGR; paper §4.1).

Indexed against the paper's appendix table (Fig. 4 -> 8). The wording, JSON
keys, and placeholder names are reproduced verbatim from the implementation
so that experiments are byte-identical to the runs reported in the paper.

Placeholders use {curly_braces}.
"""

# -----------------------------------------------------------------------------
# Fig. 4 -- Schema-guided passage grouping (Step i)
# -----------------------------------------------------------------------------

grouping_system_prompt = """For each input column, decide what kind of entity its linked passages collectively describe. That entity is the new_table_name. The source tables are given so you don't duplicate them.

- Each column gets exactly one new_table_name.
- Reuse the same new_table_name across columns whose passages describe the same kind of entity.
- new_table_name must be short snake_case.
- new_table_name must NOT match any source table name; pick a name that reflects the entity in the passages.
- The column field must use the exact form `<src>.<col>` shown in the input.

Strict JSON output:
```json
{
  "labels": [
    {
      "column": "<src>.<col>", 
      "new_table_name": "..."
    },
    ...
  ]
}
```"""


grouping_user_prompt = """Source tables (do NOT name your new tables after these):
{source_tables_block}

Linked columns and sample passages:

{linked_columns_block}"""


# -----------------------------------------------------------------------------
# Fig. 5 -- Side-table schema induction (Step ii)
# -----------------------------------------------------------------------------

schema_induction_system_prompt = """Given an entity (table_name) and sample passages, produce a list of attributes.
Each attribute's type must be one of int | float | date | text.
Prefer sortable types (int / float / date) whenever the passages support it.
Cover as much information from the passages as possible with the attributes.

Strict JSON output:
```json
{
  "attr": [
    {
      "name": "<snake_case>", 
      "type": "<int|float|date|text>"
    },
    ...
  ]
}
```"""


schema_induction_user_prompt = """table_name: {table_name}

passages:
{passages_block}"""


# -----------------------------------------------------------------------------
# Fig. 6 -- Per-table extraction prompt design (part of Step ii)
# -----------------------------------------------------------------------------

prompt_design_system_prompt = """You author the extraction-prompt content for one side table.
Inputs are the table's schema (table_name + typed attributes) and sample passages describing instances of that entity.

Produce:
- "overview": one or two short sentences describing what these passages share — the entity kind and the typical content.
- "attrs": for each schema attribute, a short phrase describing the expected value format (units, ordering, conventions) and where in a passage it usually appears.

Strict JSON output:
```json
{
  "overview": "...",
  "attrs": [
    {
      "name": "<attr_name>", 
      "format": "..."
    },
    ...
  ]
}
```"""


prompt_design_user_prompt = """table_name: {table_name}

schema:
{schema_block}

passages:
{passages_block}"""


# -----------------------------------------------------------------------------
# Fig. 7 -- Attribute-value extraction (Step iii)
# -----------------------------------------------------------------------------

extraction_system_prompt = """Extract the listed attributes from the passage.
Output JSON {attribute: value-or-null}; use null when the passage does not state it.
Extract only what the passage says about the entity indicated by cell_value.

Each attribute is given as `name (type): format`:
- `format` describes how the value appears in the passage (surface form, units, location hints) — use it to locate and read the raw value.
- `type` describes how to convert that raw value into the JSON output. Convert units / parse the surface form accordingly:
    - int   → plain integer, no thousand separators, no units.
    - float → plain number, no thousand separators, no units.
    - date  → 'YYYY-MM-DD'. Year-only → 'YYYY-01-01'; month/day-only → '9999-MM-DD'.
    - text  → short literal string copied from the passage.
Never invent values not present in the passage."""


extraction_user_prompt = """table_name: {table_name}
overview: {overview}
attrs:
{attrs_block}

cell_value: {cell_value}

passage:
{passage}"""


# -----------------------------------------------------------------------------
# Fig. 8 -- Outlier rescue (Step iv, normalization)
# -----------------------------------------------------------------------------

outlier_rescue_integer_system_prompt = """Normalize the cell to a JSON integer (no thousand separators, no units).
Return null if the cell does not state a number.
Output STRICT JSON: {"value": <int>|null}."""


outlier_rescue_real_system_prompt = """Normalize the cell to a JSON number (no thousand separators, no units).
Return null if the cell does not state a number.
Output STRICT JSON: {"value": <number>|null}."""


outlier_rescue_date_system_prompt = """Normalize the cell to a calendar date 'YYYY-MM-DD'.
Return null if the cell does not state a date.
Output STRICT JSON: {"value": "YYYY-MM-DD"|null}."""


outlier_rescue_user_prompt = """{examples_block}cell:
{cell}"""
