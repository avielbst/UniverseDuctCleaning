"""
etl/load_customers.py

Loads customers CSV into the customers table.
Idempotent: ON CONFLICT (id) DO NOTHING.
Uses batch insert via execute_values — no row-by-row iteration.
"""
import glob
import logging
import os

import numpy as np
import pandas as pd
from psycopg2.extras import execute_values

from utils import get_connection, setup_logging

logger = logging.getLogger(__name__)

# Columns we need — ignore the other 56 columns in the export
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


def _find_file(data_dir: str) -> str:
    """
    Locate the customers CSV regardless of exact filename.
    Searches data_dir for any CSV with 'customer' in the name (case-insensitive).
    Raises FileNotFoundError if nothing found.
    """
    candidates = [
        f for f in glob.glob(os.path.join(data_dir, "*.csv"))
        if "customer" in os.path.basename(f).lower()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No customer CSV found in '{data_dir}'. "
            "Expected a file with 'customer' in the filename."
        )
    if len(candidates) > 1:
        logger.warning("Multiple customer files found, using first: %s", candidates[0])
    return candidates[0]


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and normalize the raw customers dataframe.
    All transformations are vectorized — no row iteration.
    Returns a dataframe with correct types and None for missing values.
    """
    # Drop rows with no name at all — cannot identify the customer
    df = df.dropna(subset=["Display Name"])
    logger.info("After dropping null names: %d rows", len(df))
    
    # ID: strip float suffix (pandas may read as 97385658.0)
    df["ID"] = (
        df["ID"].astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )

    # Phone: exported as float (9412018793.0) → clean string
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

    # Normalize all whitespace-only strings → NaN
    str_cols = [
        "Display Name", "Mobile Number", "Email", "Lead Source",
        "Address_1 City", "Address_1 State", "Address_1 Postal Code",
    ]
    df[str_cols] = df[str_cols].replace(r"^\s*$", np.nan, regex=True)

    return df


def _build_records(df: pd.DataFrame) -> list[tuple]:
    """
    Convert cleaned dataframe into a list of tuples for execute_values.
    Uses vectorized numpy conversion to replace NaN/NaT with None.
    psycopg2 requires Python None to insert SQL NULL — not numpy NaN.
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


def load_customers(data_dir: str = "data/raw") -> int:
    """
    Locate, clean and batch-insert customers into the customers table.

    Args:
        data_dir: Directory containing the raw Jobber CSV exports.

    Returns:
        Number of rows inserted in this run.
    """
    filepath = _find_file(data_dir)
    logger.info("Loading customers from: %s", filepath)

    df = pd.read_csv(filepath, usecols=CUSTOMERS_COLS, low_memory=False)
    logger.info("Read %d rows", len(df))

    df = _clean(df)
    records = _build_records(df)

    conn = get_connection()
    with conn:
        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO customers (
                    id, name, phone, email,
                    city, state, zip,
                    lead_source, created_at
                ) VALUES %s
                ON CONFLICT (id) DO NOTHING
                RETURNING id
                """,
                records,
                page_size=500,
            )
            inserted = len(cur.fetchall())

    conn.close()
    logger.info("Inserted %d / %d rows into customers", inserted, len(df))
    return inserted


if __name__ == "__main__":
    setup_logging()
    load_customers()