"""
etl/load_jobs.py

Loads jobs into the jobs table by merging two source files:
  - data/raw/*jobs*.csv            — job metadata, status, amounts
  - data/raw/*service_request*.csv — subtotal, employee tags, finished date

Subtotal handling (three distinct cases):
  - NULL  → job has no matching service request (pre-March 2025 data)
  - 0.0   → service request exists but $0 job (free estimate)
  - > 0   → normal paid job

Idempotent: ON CONFLICT (id) DO NOTHING.
"""
import glob
import logging
import os
import re

import numpy as np
import pandas as pd
from psycopg2.extras import execute_values

from utils import clean_money, get_connection, normalize_employee_name, setup_logging

logger = logging.getLogger(__name__)


def _find_file(data_dir: str, keyword: str) -> str:
    """Find a CSV in data_dir whose filename contains keyword (case-insensitive)."""
    candidates = [
        f for f in glob.glob(os.path.join(data_dir, "*.csv"))
        if keyword.lower() in os.path.basename(f).lower()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No CSV with '{keyword}' in name found in '{data_dir}'"
        )
    if len(candidates) > 1:
        logger.warning("Multiple files for '%s', using: %s", keyword, candidates[0])
    return candidates[0]


def _build_customer_lookup(conn) -> dict:
    """Return {display_name_lower: customer_id} from the customers table."""
    with conn.cursor() as cur:
        cur.execute("SELECT id, name FROM customers")
        return {
            name.strip().lower(): cid
            for cid, name in cur.fetchall()
            if name
        }


def _build_employee_lookup(conn) -> dict:
    """Return {name_key: employee_id} from the employees table."""
    with conn.cursor() as cur:
        cur.execute("SELECT id, name_key FROM employees")
        return {name_key: eid for eid, name_key in cur.fetchall()}


def _parse_address(address) -> tuple:
    """
    Extract (city, state, zip) from an address string.
    e.g. '704 North Manatee Avenue, Arcadia, FL 34266' -> ('Arcadia', 'FL', '34266')
    Returns (None, None, None) if address is null or unparseable.
    """
    if not address or not isinstance(address, str):
        return None, None, None

    zip_match = re.search(r'\b(\d{5})(?:-\d{4})?\b', address)
    zip_code = zip_match.group(1) if zip_match else None

    parts = [p.strip() for p in address.split(',')]
    city, state = None, None
    if len(parts) >= 3:
        city = parts[-2].strip() or None
        state_zip = parts[-1].strip().split()
        state = state_zip[0] if state_zip else None

    return city, state, zip_code


def _to_none(val):
    """Convert NaN/NaT to Python None for psycopg2."""
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    return val


def load_jobs(data_dir: str = "data/raw") -> int:
    """
    Merge jobs and service_requests, resolve foreign keys, and insert into jobs.

    Args:
        data_dir: Directory containing the raw Jobber CSV exports.

    Returns:
        Number of rows inserted.
    """
    # --- Load ---
    jobs_path = _find_file(data_dir, "jobs")
    sr_path = _find_file(data_dir, "service_request")

    jobs = pd.read_csv(jobs_path, low_memory=False)
    sr = pd.read_csv(sr_path, low_memory=False)
    logger.info("Read %d jobs, %d service requests", len(jobs), len(sr))

    # --- Normalize join key ---
    jobs['job_num'] = jobs['Job #'].astype(str).str.extract(r'(\d+)')[0]
    sr['job_num'] = sr['Job #'].astype(str).str.extract(r'(\d+)')[0]

    # --- Clean money fields ---
    jobs['job_amount'] = jobs['Job amount'].apply(clean_money)

    # Subtotal: keep as float (0.0 is valid, means free estimate)
    # Do NOT coerce 0 to None — only unmatched jobs get NULL subtotal
    sr['subtotal'] = sr['Subtotal'].apply(clean_money)

    # --- Clean dates ---
    jobs['created_at'] = pd.to_datetime(
        jobs['Job created date'], utc=True, errors='coerce'
    )
    jobs['scheduled_at'] = pd.to_datetime(
        jobs['Job scheduled start date'], utc=True, errors='coerce'
    )
    sr['completed_at'] = pd.to_datetime(
        sr['Finished'], utc=True, errors='coerce'
    )

    # --- Lowercase customer name for lookup ---
    jobs['customer_name_clean'] = jobs['Customer name'].str.strip().str.lower()

    # --- Merge: jobs is the spine ---
    # Use indicator=True so we know which jobs had a matching service request
    merged = jobs.merge(
        sr[['job_num', 'subtotal', 'Job Tags', 'completed_at']],
        on='job_num',
        how='left',
        indicator=True,
    )
    logger.info("Merged into %d rows", len(merged))

    # Subtotal is NULL only when there was no service request match
    # When SR exists but subtotal=0, keep 0.0 (free estimate)
    merged['subtotal'] = merged.apply(
        lambda r: None if r['_merge'] == 'left_only' else r['subtotal'],
        axis=1,
    )
    merged.drop(columns=['_merge'], inplace=True)

    # --- Revenue fields ---
    merged['discount_amount'] = (
        merged['subtotal'].fillna(0) - merged['job_amount'].fillna(0)
    ).clip(lower=0)
    merged['discount_pct'] = np.where(
        merged['subtotal'].fillna(0) > 0,
        (merged['discount_amount'] / merged['subtotal'] * 100).round(2),
        0.0,
    )

    # --- Address parsing ---
    parsed = merged['Address'].apply(_parse_address)
    merged['city'] = [x[0] for x in parsed]
    merged['state'] = [x[1] for x in parsed]
    merged['zip'] = [x[2] for x in parsed]

    # --- FK lookups from DB ---
    conn = get_connection()
    customer_lookup = _build_customer_lookup(conn)
    employee_lookup = _build_employee_lookup(conn)

    unresolved_customers = 0
    unresolved_employees = 0

    # --- Build records ---
    records = []
    for _, row in merged.iterrows():
        # Customer FK — exact lowercase match
        cname = str(row.get('customer_name_clean') or '').strip()
        customer_id = customer_lookup.get(cname)
        if not customer_id and cname:
            unresolved_customers += 1

        # Employee FK — extract from job tags
        tech_name = normalize_employee_name(row.get('Job Tags'))
        employee_id = employee_lookup.get(tech_name) if tech_name else None
        if tech_name and not employee_id:
            unresolved_employees += 1

        records.append((
            row['job_num'],
            customer_id,
            employee_id,
            _to_none(row.get('Job status')),
            _to_none(row.get('Job description')),
            _to_none(row.get('Address')),
            _to_none(row.get('city')),
            _to_none(row.get('state')),
            _to_none(row.get('zip')),
            _to_none(row.get('job_amount')),
            _to_none(row.get('subtotal')),
            _to_none(row.get('discount_amount')),
            _to_none(row.get('discount_pct')),
            _to_none(row.get('scheduled_at')),
            _to_none(row.get('completed_at')),
            _to_none(row.get('created_at')),
            _to_none(row.get('Notes')),
        ))

    # --- Batch insert ---
    with conn:
        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO jobs (
                    id, customer_id, employee_id,
                    status, description,
                    address, city, state, zip,
                    job_amount, subtotal, discount_amount, discount_pct,
                    scheduled_at, completed_at, created_at,
                    notes
                ) VALUES %s
                ON CONFLICT (id) DO NOTHING
                RETURNING id
                """,
                records,
                page_size=500,
            )
            inserted = len(cur.fetchall())

    conn.close()

    logger.info("Inserted %d / %d rows into jobs", inserted, len(merged))
    if unresolved_customers:
        logger.warning(
            "%d jobs with unresolved customer names — customer_id set to NULL",
            unresolved_customers,
        )
    if unresolved_employees:
        logger.warning(
            "%d jobs with unresolved employee tags — employee_id set to NULL "
            "(employee may have been removed from employees.csv)",
            unresolved_employees,
        )
    return inserted


if __name__ == "__main__":
    setup_logging()
    load_jobs()