"""Question-Conditioned Relational Completion (QCRC) -- TRACE paper §4.2.

(i)   planning.py        question-conditioned planning              (Fig. 9)
(ii)  completion.py      embedding-similarity gate + targeted       (Fig. 10)
(iii)                    extraction -- produces D_q
(iv)  agent.py           SQL execution over completed state         (Fig. 11)
(v)                      + residual-evidence fallback (only when SQL returns 0 rows)
"""
