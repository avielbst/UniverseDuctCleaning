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


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and normalize the employees dataframe.
    Ensures name_key is uppercase and all numeric columns are properly typed.
    """
    # name_key must be uppercase for matching against job tags
    df["name_key"] = df["name_key"].str.strip().str.upper()
    df["name"] = df["name"].str.strip()

    # All numeric columns — coerce errors to NaN
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

    # Convert NaN → None for psycopg2
    df = df.astype(object).where(df.notna(), other=None)

    return df


def load_employees(filepath: str = "data/employees.csv") -> int:
    """
    Load employees from the manually maintained CSV into the employees table.

    Args:
        filepath: Path to the employees CSV file.

    Returns:
        Number of new rows inserted.
    """
    df = pd.read_csv(filepath)
    logger.info("Read %d employees from %s", len(df), filepath)

    df = _clean(df)

    records = list(
        df[EMPLOYEE_COLS].itertuples(index=False, name=None)
    )

    conn = get_connection()
    with conn:
        with conn.cursor() as cur:
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
                RETURNING id
                """,
                records,
            )
            inserted = len(cur.fetchall())

    conn.close()
    logger.info("Inserted %d / %d rows into employees", inserted, len(df))
    return inserted


if __name__ == "__main__":
    setup_logging()
    load_employees()