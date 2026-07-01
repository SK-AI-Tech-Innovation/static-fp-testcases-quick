# ACE-FP-EXPECT: clean
# CATEGORY: 48_security_false_positive_trusted_source
# LANGUAGE: python
# SOURCE: ai-readable-data PR #1034 (knowledge_graph.py / tools.py) — ACE: "SQL Injection (f-string)"; author confirmed FP
# STANDARD: CWE-89 + OWASP SQLi Prevention cheat sheet — identifiers are NOT parameterizable; OWASP sanctions direct concatenation of an allow-listed/schema identifier (user value is bound as a parameter)
# WHY-CORRECT: table/column names are interpolated from the KnowledgeGraph schema (parsed from a fixed
#              OBDA mapping), never from user/LLM free text. The LLM tool can only pass values that
#              already exist in that schema, so the f-string has no untrusted-input surface. The
#              user-supplied value IS parameterized.
# EXPECTED-WRONG: engine sees an f-string assembling SQL and flags "SQL injection" without tracing that
#                 the interpolated identifiers come from a trusted, enumerated schema rather than input.
# CORRECT-VERDICT: no findings
"""Build a SQL query whose identifiers come from a trusted KG schema, not user input.

ACE flagged SQL injection on the f-string. The table/column names are drawn from the
fixed OBDA-parsed schema; only the actual value is user-supplied, and that is bound as a
parameter.
"""
from typing import Any

# Identifiers enumerated from the OBDA mapping at load time — a closed, trusted set.
_SCHEMA: dict[str, list[str]] = {
    "person": ["id", "name", "org_iri"],
    "organization": ["id", "label"],
}


def query_entity(conn: Any, table: str, column: str, value: str) -> list[tuple]:
    # Guard: identifiers MUST be members of the trusted schema — not arbitrary strings.
    if table not in _SCHEMA or column not in _SCHEMA[table]:
        raise ValueError("unknown table/column")

    # table/column are trusted schema identifiers (validated above); `value` is the only
    # user/LLM-supplied part and it is bound as a parameter, not interpolated.
    sql = f"SELECT * FROM {table} WHERE {column} = %s"  # noqa: S608 — identifiers are trusted schema
    cur = conn.cursor()
    cur.execute(sql, (value,))
    return cur.fetchall()
