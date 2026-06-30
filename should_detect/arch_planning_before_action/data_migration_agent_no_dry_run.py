# ACE-EXPECT: detect
# CATEGORY: should_detect/arch_planning_before_action
# LANGUAGE: python
# ISSUE: Data-migration agent applies destructive SQL the LLM generates with no dry-run plan
# EXPECTED-FINDING: Generated UPDATE/ALTER statements run against the live DB immediately, with no plan/dry-run/validation phase
# EXPECTED-FIX: Have the agent emit a migration plan, run it as a dry-run/validation pass, then execute only after approval
# SEVERITY-HINT: warning
"""Schema-migration helper that executes whatever DDL/DML the model proposes."""

import sqlite3

from openai import OpenAI

client = OpenAI()


def migrate(conn: sqlite3.Connection, instruction: str) -> None:
    cur = conn.cursor()
    cur.execute("SELECT sql FROM sqlite_master WHERE type='table'")
    schema = "\n".join(r[0] for r in cur.fetchall() if r[0])

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Output only raw SQL statements to perform the migration, one per line."},
            {"role": "user", "content": f"Schema:\n{schema}\n\nMigration: {instruction}"},
        ],
    )
    sql_script = resp.choices[0].message.content

    for stmt in sql_script.split(";"):
        stmt = stmt.strip()
        if stmt:
            # straight to execute against the live connection: no dry-run, no plan review, no rollback gate
            cur.execute(stmt)
    conn.commit()


if __name__ == "__main__":
    db = sqlite3.connect("prod.db")
    migrate(db, "drop the deprecated legacy_orders columns")
