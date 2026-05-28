"""Schema-Guided Relationalization (SGR) -- TRACE paper §4.1.

(i)   grouping.py            schema-guided passage grouping        (Fig. 4)
(ii)  schema_induction.py    side-table schema induction           (Fig. 5)
      prompt_design.py       per-table extraction prompt design    (Fig. 6)
(iii) extraction.py          attribute-value extraction            (Fig. 7)
(iv)  source_normalize.py    source-table normalization            (algo + Fig. 8 rescue, HybridQA only)
      normalization.py       side-table column re-typing           (Fig. 8 outlier rescue)
      assembly.py            relational database assembly          -> D_m
"""
