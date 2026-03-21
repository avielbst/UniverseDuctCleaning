# etl/utils.py
import os
import logging
from dotenv import load_dotenv
import psycopg2

load_dotenv()

def get_connection():
    """Return a psycopg2 connection using credentials from .env"""
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", 5432)),
        dbname=os.getenv("DB_NAME", "ductai_db"),
        user=os.getenv("DB_USER", "ductai"),
        password=os.getenv("DB_PASSWORD", "ductai"),
    )

def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("etl.log"),
            logging.StreamHandler()
        ]
    )

def clean_money(value):
    """Convert money string to float, e.g. '$1,234.56' -> 1234.56"""
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip().replace('$', '').replace(',', '')
        if value == '':
            return None
        try:
            return float(value)
        except ValueError:
            return None
    return float(value)

def normalize_service_name(name: str | None) -> str | None:
    """Normalize service name to canonical lowercase form.
    
    Examples:
        "Dryer Vent Cleaning "        -> "dryer vent cleaning"
        "AIR DUCT DEEP CLEANING"      -> "air duct deep cleaning"
        "dryer vent routine cleaning" -> "dryer vent cleaning"
    """
    if not name:
        return None
    
    s = name.strip().lower()
    
    # Canonical mappings for known variants found in audit
    mappings = {
        "dryer vent routine cleaning":          "dryer vent cleaning",
        "air duct maintenance cleaning":        "maintenance air duct cleaning",
        "maintenance cleaning":                 "maintenance air duct cleaning",
        "maintenance duct cleaning":            "maintenance air duct cleaning",
        "air duct cleaning":                    "air duct deep cleaning",
        "duct and vent cleaning":               "air duct deep cleaning",
        "supply vent deep cleaning":            "air duct deep cleaning",
        "blower deep cleaning + coil maintenance": "blower deep cleaning and coil maintenance",
        "blower deep cleaning & disinfectant":  "blower deep cleaning and disinfectant",
    }
    
    return mappings.get(s, s)  # If not found in mappings, return the cleaned string as is

import re

def normalize_employee_name(tag: str | None) -> str | None:
    """Extract and normalize technician name from a job tag string.
    
    Examples:
        "Technician: MICHAEL"           -> "MICHAEL"
        "Technician : MATTHEW"          -> "MATTHEW"  
        "Free Estimate,Technician: ROY" -> "ROY"
        "Technician: VIVO&DAVID"        -> "VIVO"
        "Free Estimate"                 -> None
        None                            -> None
    """
    if not tag:
        return None
    
    match = re.search(r'[Tt]echnician\s*:\s*([A-Z][A-Z &]+)', tag)
    if not match:
        return None
    
    name = match.group(1).strip()
    # If multiple technicians (e.g. "VIVO&DAVID"), take the first
    name = re.split(r'[&,]', name)[0].strip()
    return name.upper()