"""
etl/load_employees.py

Loads employees from the manually maintained data/employees.csv
into the employees table.
Idempotent: ON CONFLICT (name_key) DO NOTHING.

The owner maintains data/employees.csv directly in Excel to update
commission rates, add new employees, or change pay types.
Re-running this loader picks up any new rows without touching existing ones.
"""
import logging

import pandas as pd
from psycopg2.extras import execute_values

from utils import get_connection, setup_logging

logger = logging.getLogger(__name__)

EMPLOYEE_COLS = [
    "name",
    "name_key",
    "role",
    "pay_type",
    "commission_rate",
    "commission_tier1_rate",
    "commission_tier1_threshold",
    "commission_tier2_rate",
    "commission_tier2_threshold",
    "hourly_rate",
    "monthly_salary",
]

# Employees CSV is manually maintained outside data/raw — committed to repo
EMPLOYEES_CSV = "data/employees.csv"


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize the employees dataframe."""
    df["name_key"] = df["name_key"].str.strip().str.upper()
    df["name"] = df["name"].str.strip()

    numeric_cols = [
        "commission_rate",
        "commission_tier1_rate",
        "commission_tier1_threshold",
        "commission_tier2_rate",
        "commission_tier2_threshold",
        "hourly_rate",
        "monthly_salary",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.astype(object).where(df.notna(), other=None)
    return df


def load_employees(data_dir: str = "data/raw") -> int:
    """
    Load employees from the manually maintained CSV into the employees table.
    The data_dir argument is accepted for interface consistency with other
    loaders but is not used — employees.csv lives at a fixed committed path.

    Args:
        data_dir: Unused. Kept for consistent loader interface with run_all.

    Returns:
        Number of new rows inserted.
    """
    df = pd.read_csv(EMPLOYEES_CSV)
    logger.info("Read %d employees from %s", len(df), EMPLOYEES_CSV)

    df = _clean(df)
    records = list(df[EMPLOYEE_COLS].itertuples(index=False, name=None))

    conn = get_connection()
    with conn:
        with conn.cursor() as cur:
            # Use before/after count — ON CONFLICT DO NOTHING makes rowcount unreliable
            cur.execute("SELECT COUNT(*) FROM employees")
            before = cur.fetchone()[0]

            execute_values(
                cur,
                """
                INSERT INTO employees (
                    name, name_key, role, pay_type,
                    commission_rate,
                    commission_tier1_rate, commission_tier1_threshold,
                    commission_tier2_rate, commission_tier2_threshold,
                    hourly_rate, monthly_salary
                ) VALUES %s
                ON CONFLICT (name_key) DO NOTHING
                """,
                records,
            )

            cur.execute("SELECT COUNT(*) FROM employees")
            after = cur.fetchone()[0]

    conn.close()
    inserted = after - before
    logger.info("Inserted %d / %d rows into employees", inserted, len(df))
    return inserted


if __name__ == "__main__":
    setup_logging()
    load_employees()
