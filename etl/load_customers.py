"""
Loads customers CSV into the customers table.
Idempotent: ON CONFLICT (id) DO NOTHING.
Uses batch insert.
"""
import logging
import numpy as np
import pandas as pd
from psycopg2.extras import execute_values

from utils import get_connection, setup_logging, find_file

logger = logging.getLogger(__name__)

# Relevant columns
CUSTOMERS_COLS = [
    "ID",
    "Display Name",
    "Mobile Number",
    "Email",
    "Lead Source",
    "Address_1 City",
    "Address_1 State",
    "Address_1 Postal Code",
    "Customer created at",
]


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and normalize the raw customers dataframe.
    """
    # Drop rows with no name at all - likely invalid.
    df = df.dropna(subset=["Display Name"])
    logger.info("After dropping null names: %d rows", len(df))
    
    # ID: strip float suffix (x.0)
    df["ID"] = (
        df["ID"].astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )

    # Phone: convert to clean string
    df["Mobile Number"] = (
        df["Mobile Number"]
        .fillna("")
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )

    # Postal code: same float issue
    df["Address_1 Postal Code"] = (
        df["Address_1 Postal Code"]
        .fillna("")
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )

    # Dates
    df["Customer created at"] = pd.to_datetime(
        df["Customer created at"], utc=True, errors="coerce"
    )

    # Normalize all whitespace-only strings to NaN
    str_cols = [
        "Display Name", "Mobile Number", "Email", "Lead Source",
        "Address_1 City", "Address_1 State", "Address_1 Postal Code",
    ]
    df[str_cols] = df[str_cols].replace(r"^\s*$", np.nan, regex=True)

    return df


def _build_records(df: pd.DataFrame) -> list[tuple]:
    """
    Convert cleaned dataframe into a list of tuples for insertion.
    Uses vectorized numpy conversion to replace NaN/NaT with None.
    psycopg2 requires Python None to insert SQL NULL - not numpy NaN.
    """
    # Select and rename columns in the exact insert order
    ordered = df[[
        "ID",
        "Display Name",
        "Mobile Number",
        "Email",
        "Address_1 City",
        "Address_1 State",
        "Address_1 Postal Code",
        "Lead Source",
        "Customer created at",
    ]]

    # Convert to object dtype so NaN becomes None uniformly
    # numpy where trick: replace NaN with None across all columns at once
    arr = ordered.astype(object).where(ordered.notna(), other=None)

    return list(arr.itertuples(index=False, name=None))


def load_customers(path: str = "data/raw") -> int:
    """
    Locate, clean and batch-insert customers into the customers table.

    Args:
        path: Either a specific CSV filepath (when called from run_all)
              or a data directory (when called directly).

    Returns:
        Number of rows inserted in this run.
    """
    filepath = path if path.endswith(".csv") else find_file(path, "customer")

    logger.info("Loading customers from: %s", filepath)

    df = pd.read_csv(filepath, usecols=CUSTOMERS_COLS, low_memory=False)
    logger.info("Read %d rows", len(df))

    df = _clean(df)
    records = _build_records(df)

    conn = get_connection()
    with conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM customers")
            before = cur.fetchone()[0]

            execute_values(
                cur,
                """
                INSERT INTO customers (
                    id, name, phone, email,
                    city, state, zip,
                    lead_source, created_at
                ) VALUES %s
                ON CONFLICT (id) DO NOTHING
                """,
                records,
                page_size=500,
            )

            cur.execute("SELECT COUNT(*) FROM customers")
            after = cur.fetchone()[0]
            inserted = after - before

    conn.close()
    logger.info("Inserted %d / %d rows into customers", inserted, len(df))
    return inserted


if __name__ == "__main__":
    setup_logging()
    load_customers()