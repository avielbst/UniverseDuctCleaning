"""

Single entrypoint for the full ETL pipeline.
Designed to run nightly - processes all raw CSV files in data/raw/,
moves successfully processed files to data/processed/YYYY-MM-DD/,
and tracks failures by renaming files with a __failures_N suffix.

File lifecycle:
  data/raw/customers.csv              # fresh file
  data/raw/customers__failures_1.csv  # failed once
  data/raw/customers__failures_2.csv  # failed twice
  data/raw/customers__failures_3.csv  # -> moved to data/failed/
  data/processed/2026-03-22/customers.csv  # successfully processed

Loader execution order (respects FK dependencies):
  1. customers   - no FK deps
  2. employees   - no FK deps
  3. jobs        - requires customers + employees
  4. line_items  - requires jobs
  5. estimates   - requires customers

Usage:
  uv run ./etl/run_all.py
  uv run ./etl/run_all.py --data-dir data/raw --dry-run
"""
import argparse
import logging
import os
import re
import shutil
from datetime import date
from typing import Callable

from utils import setup_logging

logger = logging.getLogger(__name__)

MAX_FAILURES = 3
PROCESSED_BASE = "data/processed"
FAILED_DIR = "data/failed"


# ---------------------------------------------------------------------------
# File management helpers
# ---------------------------------------------------------------------------

def _get_failure_count(filepath: str) -> int:
    """
    Extract failure count from filename suffix.
    e.g. 'customers__failures_2.csv' -> 2
         'customers.csv' -> 0
    """
    name = os.path.basename(filepath)
    match = re.search(r"__failures_(\d+)\.", name)
    return int(match.group(1)) if match else 0


def _increment_failure(filepath: str) -> str:
    """
    Rename file to increment its failure counter.
    Returns the new filepath.

    customers.csv           -> customers__failures_1.csv
    customers__failures_1.csv -> customers__failures_2.csv
    """
    dirpath = os.path.dirname(filepath)
    name = os.path.basename(filepath)
    count = _get_failure_count(filepath)
    new_count = count + 1

    if count == 0:
        # Insert failure suffix before extension
        base, ext = os.path.splitext(name)
        new_name = f"{base}__failures_{new_count}{ext}"
    else:
        # Replace existing failure count
        new_name = re.sub(
            r"__failures_\d+(\.\w+)$",
            f"__failures_{new_count}\\1",
            name,
        )

    new_path = os.path.join(dirpath, new_name)
    os.rename(filepath, new_path)
    logger.warning("Failure recorded - renamed to: %s", new_name)
    return new_path


def _move_to_processed(filepath: str) -> None:
    """
    Move a successfully processed file to data/processed/YYYY-MM-DD/.
    Strips any __failures_N suffix from the destination filename.
    """
    today = date.today().isoformat()
    dest_dir = os.path.join(PROCESSED_BASE, today)
    os.makedirs(dest_dir, exist_ok=True)

    name = os.path.basename(filepath)
    # Strip failure suffix for clean archive name
    clean_name = re.sub(r"__failures_\d+(\.\w+)$", r"\1", name)
    dest_path = os.path.join(dest_dir, clean_name)

    shutil.move(filepath, dest_path)
    logger.info("Moved to processed: %s", dest_path)


def _move_to_failed(filepath: str) -> None:
    """
    Move a file that has exceeded MAX_FAILURES to data/failed/.
    """
    os.makedirs(FAILED_DIR, exist_ok=True)
    dest_path = os.path.join(FAILED_DIR, os.path.basename(filepath))
    shutil.move(filepath, dest_path)
    logger.error(
        "File exceeded %d failures - moved to failed/: %s",
        MAX_FAILURES,
        os.path.basename(filepath),
    )


# ---------------------------------------------------------------------------
# Loader wrapper
# ---------------------------------------------------------------------------

def _run_loader(
    keyword: str,
    loader_fn: Callable[[str], int],
    data_dir: str,
    dry_run: bool = False,
) -> None:
    """
    Find all CSVs matching keyword, run loader_fn on each file explicitly,
    and handle success/failure file management.

    Each matching file is passed directly to loader_fn so the loader
    processes exactly that file - not whatever find_file() returns internally.
    This prevents the double-processing bug where find_file() always returns
    the first match regardless of which file the orchestrator is iterating.

    Special case: 'employee' keyword - employees.csv lives at a fixed path
    outside data/raw and is called once regardless of file matches in data_dir.
    """
    import glob

    # employees.csv is not in data/raw - handle as special case
    if keyword == "employee":
        if dry_run:
            logger.info("[DRY RUN] Would run employee loader")
            return
        try:
            inserted = loader_fn(data_dir)
            logger.info("Loader 'employee' completed - %d rows inserted", inserted)
        except Exception as e:
            logger.error("Loader 'employee' failed: %s", e)
        return

    # Find all matching files including those with failure suffixes
    all_files = glob.glob(os.path.join(data_dir, "*.csv"))
    matches = [
        f for f in all_files
        if keyword.lower() in os.path.basename(f).lower().split("__failures")[0]
    ]

    if not matches:
        logger.warning("No CSV found matching keyword '%s' in %s", keyword, data_dir)
        return

    for filepath in sorted(matches):
        name = os.path.basename(filepath)
        failures = _get_failure_count(filepath)

        logger.info("Processing: %s (previous failures: %d)", name, failures)

        if dry_run:
            logger.info("[DRY RUN] Would process: %s", name)
            continue

        try:
            # Pass specific filepath so loader processes exactly this file
            inserted = loader_fn(filepath)
            logger.info(
                "Loader '%s' completed - %d rows inserted", keyword, inserted
            )
            _move_to_processed(filepath)

        except Exception as e:
            logger.error("Loader '%s' failed on %s: %s", keyword, name, e)
            new_failures = failures + 1

            if new_failures >= MAX_FAILURES:
                _move_to_failed(filepath)
            else:
                _increment_failure(filepath)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_all(data_dir: str = "data/raw", dry_run: bool = False) -> None:
    """
    Run the full ETL pipeline in FK-dependency order.

    Args:
        data_dir: Directory containing raw CSV exports.
        dry_run:  If True, log what would happen without executing.
    """
    # Import loaders here to avoid circular imports and keep startup fast
    from load_customers import load_customers
    from load_employees import load_employees
    from load_jobs import load_jobs
    from load_line_items import load_line_items
    from load_estimates import load_estimates

    os.makedirs(PROCESSED_BASE, exist_ok=True)
    os.makedirs(FAILED_DIR, exist_ok=True)

    logger.info("=" * 60)
    logger.info("DuctAI ETL pipeline starting%s", " [DRY RUN]" if dry_run else "")
    logger.info("Data directory: %s", data_dir)
    logger.info("=" * 60)

    # Execution order matters - do not change
    pipeline = [
        ("customer",        load_customers),   # 1. No FK deps
        ("employee",        load_employees),   # 2. No FK deps
        ("jobs",            load_jobs),        # 3. Requires customers + employees
        ("service_request", load_line_items),  # 4. Requires jobs
        ("estimate",        load_estimates),   # 5. Requires customers
    ]

    results = {}
    for keyword, loader_fn in pipeline:
        logger.info("-" * 40)
        logger.info("Stage: %s", keyword)
        try:
            _run_loader(keyword, loader_fn, data_dir, dry_run)
            results[keyword] = "success"
        except Exception as e:
            logger.error("Unexpected error in stage '%s': %s", keyword, e)
            results[keyword] = f"error: {e}"

    # --- Summary ---
    logger.info("=" * 60)
    logger.info("Pipeline complete. Summary:")
    for stage, status in results.items():
        icon = "SUCCESS" if status == "success" else "FAILURE"
        logger.info("  %s %s - %s", icon, stage, status)
    logger.info("=" * 60)


if __name__ == "__main__":
    setup_logging()

    parser = argparse.ArgumentParser(
        description="DuctAI ETL pipeline - loads all CRM exports into PostgreSQL"
    )
    parser.add_argument(
        "--data-dir",
        default="data/raw",
        help="Directory containing raw CSV exports (default: data/raw)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log what would happen without executing DB inserts or file moves",
    )
    args = parser.parse_args()

    run_all(data_dir=args.data_dir, dry_run=args.dry_run)