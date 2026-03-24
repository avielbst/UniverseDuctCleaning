"""
sql_tool — LangChain tool for safe read-only SQL queries.

Only SELECT statements are allowed. All other SQL is blocked before execution.
Results are capped at 100 rows.
"""
import json
import re

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from etl.utils import get_connection

_BLOCKED_PATTERNS = re.compile(
    r"\b(drop|insert|update|delete|truncate|create|alter|grant|revoke|execute|exec|pg_)\b",
    re.IGNORECASE,
)

_LIMIT_RE = re.compile(r"\blimit\s+\d+", re.IGNORECASE)


def _safe_query(query: str) -> str:
    """Validate and execute a SELECT-only query; return results as a JSON string."""
    stripped = query.strip()

    # Must start with SELECT
    first_word = stripped.split()[0].lower() if stripped else ""
    if first_word != "select":
        return "Error: only SELECT queries are allowed."

    # Block dangerous substrings (catches inline comments, DDL in subqueries, etc.)
    if "--" in stripped or ";" in stripped:
        return "Error: query contains disallowed characters (-- or ;)."
    if _BLOCKED_PATTERNS.search(stripped):
        return "Error: query contains a disallowed keyword."

    # Enforce row cap
    if not _LIMIT_RE.search(stripped):
        stripped = stripped.rstrip() + " LIMIT 100"

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(stripped)
                columns = [desc[0] for desc in cur.description]
                rows = [dict(zip(columns, row)) for row in cur.fetchall()]
        return json.dumps(rows, indent=2, default=str)
    except Exception as e:
        return f"SQL error: {e}"


class SqlInput(BaseModel):
    query: str = Field(description="A read-only SELECT SQL query against the DuctAI database")


@tool("sql_tool", args_schema=SqlInput)
def sql_tool(query: str) -> str:
    """
    Execute a read-only SQL SELECT query against the DuctAI PostgreSQL database.
    Use this to answer business questions about jobs, revenue, customers, employees,
    estimates, and service mix. The database has these analytical views:
      v_service_mix, v_revenue_summary, v_employee_performance,
      v_new_customers_per_month, v_region_summary, v_lead_source_roi,
      v_estimate_pipeline, v_customer_retention, v_service_cooccurrence,
      v_customer_identity, v_customer_history, v_city_price_index.
    Only SELECT statements are permitted. Results are capped at 100 rows.
    """
    return _safe_query(query)
