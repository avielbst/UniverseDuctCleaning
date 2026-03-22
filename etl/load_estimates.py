"""
etl/load_estimates.py

Loads estimates CSV into the estimates table.
Idempotent: ON CONFLICT (id) DO NOTHING.
Uses batch insert.

Value logic: Jobber stores won/open/lost values in three separate columns
but only one is populated per row based on outcome. Consolidated into
a single `value` column:
  outcome='won'  -> value = Won value
  outcome='lost' -> value = Lost value
  outcome='open' -> value = Open value
"""
import logging

import numpy as np
import pandas as pd
from psycopg2.extras import execute_values

from utils import clean_money, find_file, get_connection, setup_logging

logger = logging.getLogger(__name__)

ESTIMATES_COLS = [
    "Estimate #",
    "Customer name",
    "Estimate status",
    "Outcome",
    "Won value",
    "Open value",
    "Lost value",
    "Estimate lead source",
    "Created date",
    "Scheduled date",
]


def _build_customer_lookup(conn) -> dict:
    """Return {display_name_lower: customer_id} from the customers table."""
    with conn.cursor() as cur:
        cur.execute("SELECT id, name FROM customers")
        return {
            name.strip().lower(): cid
            for cid, name in cur.fetchall()
            if name
        }


def _resolve_value(row: pd.Series) -> float | None:
    """
    Consolidate the three outcome value columns into one.
    Only one column is populated per row — pick the relevant one.
    """
    outcome = str(row.get("Outcome", "")).strip().lower()
    if outcome == "won":
        return clean_money(row.get("Won value"))
    elif outcome == "lost":
        return clean_money(row.get("Lost value"))
    elif outcome == "open":
        return clean_money(row.get("Open value"))
    return None


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize the estimates dataframe."""
    # Dates
    df["Created date"] = pd.to_datetime(
        df["Created date"], utc=True, errors="coerce"
    )
    df["Scheduled date"] = pd.to_datetime(
        df["Scheduled date"], utc=True, errors="coerce"
    )

    # Consolidated value column
    df["value"] = df.apply(_resolve_value, axis=1)

    # Lowercase customer name for lookup
    df["customer_name_clean"] = df["Customer name"].str.strip().str.lower()

    # Normalize empty strings to NaN across string columns
    str_cols = ["Estimate status", "Outcome", "Estimate lead source"]
    df[str_cols] = df[str_cols].replace(r"^\s*$", np.nan, regex=True)

    return df


def _build_records(df: pd.DataFrame, customer_lookup: dict) -> tuple[list, int]:
    """
    Build list of tuples for execute_values.
    Returns (records, unresolved_count).
    """
    unresolved = 0

    def to_none(val):
        try:
            if pd.isna(val):
                return None
        except (TypeError, ValueError):
            pass
        return val

    records = []
    for _, row in df.iterrows():
        customer_id = customer_lookup.get(row["customer_name_clean"])
        if not customer_id:
            unresolved += 1

        records.append((
            str(int(row["Estimate #"])),
            customer_id,
            to_none(row.get("Estimate status")),
            to_none(row.get("Outcome")),
            to_none(row.get("value")),
            to_none(row.get("Estimate lead source")),
            to_none(row.get("Created date")),
            to_none(row.get("Scheduled date")),
        ))

    return records, unresolved


def load_estimates(path: str = "data/raw") -> int:
    """
    Locate, clean and batch-insert estimates into the estimates table.

    Args:
        path: Either a specific CSV filepath (when called from run_all)
              or a data directory (when called directly).

    Returns:
        Number of rows inserted in this run.
    """
    filepath = path if path.endswith(".csv") else find_file(path, "estimates")
    logger.info("Loading estimates from: %s", filepath)

    df = pd.read_csv(filepath, usecols=ESTIMATES_COLS, low_memory=False)
    logger.info("Read %d rows", len(df))

    df = _clean(df)

    conn = get_connection()
    customer_lookup = _build_customer_lookup(conn)
    records, unresolved = _build_records(df, customer_lookup)

    with conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM estimates")
            before = cur.fetchone()[0]

            execute_values(
                cur,
                """
                INSERT INTO estimates (
                    id, customer_id,
                    status, outcome, value,
                    lead_source, created_at, scheduled_at
                ) VALUES %s
                ON CONFLICT (id) DO NOTHING
                """,
                records,
                page_size=500,
            )

            cur.execute("SELECT COUNT(*) FROM estimates")
            after = cur.fetchone()[0]
            inserted = after - before

    conn.close()
    logger.info("Inserted %d / %d rows into estimates", inserted, len(df))
    if unresolved:
        logger.warning(
            "%d estimates had unresolved customer names — customer_id set to NULL",
            unresolved,
        )
    return inserted


if __name__ == "__main__":
    setup_logging()
    load_estimates()