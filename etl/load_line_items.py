"""
Parses the Line Items text column from service_requests CSV into
individual rows in the line_items table.

Raw format per job:
    SERVICES
    Sanitation Disinfectant - $367.00
    Blower Cleaning - $299.00
    Maintenance Air Duct Cleaning - $199.00

Each parsed service becomes one row in line_items.
Idempotent: truncates and reloads (no natural PK on raw line items).
"""
import glob
import logging
import os
import re

import pandas as pd
from psycopg2.extras import execute_values

from utils import get_connection, normalize_service_name, setup_logging, find_file

logger = logging.getLogger(__name__)

# Regex: capture service name and price
# Non-greedy (.+?) so "Duct Encapsulation - FIBERGLASS - $2235.00"
# captures "Duct Encapsulation - FIBERGLASS" not just "Duct Encapsulation"
LINE_PATTERN = re.compile(r'^(.+?)\s*-\s*\$([\d,]+\.?\d*)$')

# Lines to skip entirely
SKIP_LINES = {'services', 'materials'}


def _parse_line_items(job_id: str, raw_text: str) -> list[tuple]:
    """
    Parse a raw line items text blob into a list of tuples ready for insert.

    Args:
        job_id:   The job ID this text belongs to.
        raw_text: The raw multi-line string from the Line Items column.

    Returns:
        List of (job_id, service, service_key, price) tuples.
        Empty list if no valid lines found.
    """
    records = []

    for line in re.split(r'[\r\n]+', raw_text):
        line = line.strip()

        # Skip empty lines and section headers
        if not line or line.lower() in SKIP_LINES:
            continue

        # Skip bundle totals - these duplicate individual line item values
        if 'bundle total' in line.lower() or 'package bundle' in line.lower():
            continue

        match = LINE_PATTERN.match(line)
        if not match:
            # Line exists but has no price - skip silently
            logger.debug("No price found, skipping line: '%s'", line)
            continue

        service_raw = match.group(1).strip()
        price_str = match.group(2).replace(',', '')

        try:
            price = float(price_str)
        except ValueError:
            logger.warning("Could not parse price '%s' on line: '%s'", price_str, line)
            continue

        service_key = normalize_service_name(service_raw)

        records.append((job_id, service_raw, service_key, price))

    return records


def load_line_items(path: str = "data/raw") -> int:
    """
    Parse and batch-insert all line items from service_requests CSV.

    Strategy: truncate then reload.
    Line items have no natural PK - they are derived rows.

    Args:
        path: Either a specific CSV filepath (when called from run_all)
              or a data directory (when called directly).

    Returns:
        Total number of rows inserted.
    """
    sr_path = path if path.endswith(".csv") else find_file(path, "service_request")
    sr = pd.read_csv(sr_path, low_memory=False)
    logger.info("Read %d service request rows from %s", len(sr), sr_path)

    # Normalize job number - used as FK into jobs table
    sr['job_id'] = sr['Job #'].astype(str).str.extract(r'(\d+)')[0]

    # Only process rows that have line items
    has_items = sr[sr['Line Items'].notna()].copy()
    logger.info("%d / %d rows have line items", len(has_items), len(sr))

    # --- Parse all line items ---
    all_records = []
    skipped_jobs = 0

    for _, row in has_items.iterrows():
        job_id = row['job_id']
        parsed = _parse_line_items(job_id, str(row['Line Items']))
        if parsed:
            all_records.extend(parsed)
        else:
            skipped_jobs += 1

    logger.info(
        "Parsed %d line item rows from %d jobs (%d jobs had no parseable items)",
        len(all_records), len(has_items), skipped_jobs
    )

    # --- Filter to only jobs that exist in the jobs table ---
    conn = get_connection()
    with conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM jobs")
            valid_job_ids = {row[0] for row in cur.fetchall()}

        before = len(all_records)
        all_records = [r for r in all_records if r[0] in valid_job_ids]
        filtered = before - len(all_records)
        if filtered:
            logger.warning(
                "Dropped %d line items referencing job IDs not in jobs table "
                "(likely timing gap between exports)",
                filtered,
            )

        if not all_records:
            logger.warning("No line items to insert - check source data")
            conn.close()
            return 0

        # --- Truncate and reload ---
        with conn.cursor() as cur:
            cur.execute("TRUNCATE line_items RESTART IDENTITY")
            execute_values(
                cur,
                """
                INSERT INTO line_items (job_id, service, service_key, price)
                VALUES %s
                """,
                all_records,
                page_size=1000,
            )
            cur.execute("SELECT COUNT(*) FROM line_items")
            inserted = cur.fetchone()[0]

    conn.close()
    logger.info("Inserted %d rows into line_items", inserted)
    return inserted


if __name__ == "__main__":
    setup_logging()
    load_line_items()