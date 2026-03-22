# etl/utils.py
import os
import logging
from dotenv import load_dotenv
import psycopg2
import re
import glob

logger = logging.getLogger(__name__)

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

def find_file(data_dir: str, keyword: str) -> str:
    candidates = [
        f for f in glob.glob(os.path.join(data_dir, "*.csv"))
        if keyword.lower() in os.path.basename(f).lower().split("__failures")[0]
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No CSV with '{keyword}' in name found in '{data_dir}'"
        )
    if len(candidates) > 1:
        logger.warning("Multiple files for '%s', using: %s", keyword, candidates[0])
    return candidates[0]

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
    if not name:
        return None

    s = name.strip().lower()

    mappings = {
        # --- Dryer vent ---
        "dryer vent routine cleaning":              "dryer vent cleaning",
        "dryer vent deep cleaning":                 "dryer vent cleaning",
        "dryer cleaning":                           "dryer vent cleaning",
        "dryer vent cleaning + roof unclog":        "dryer vent cleaning roof unclog",
        "dryer vent cleaning - roof unclog":        "dryer vent cleaning roof unclog",
        "dryer vent roof unclog":                   "dryer vent cleaning roof unclog",
        "dryer vent cleaning - attic unclog":       "dryer vent cleaning attic unclog",
        "dryer vent cleaning additional length":    "dryer vent additional length",

        # --- Air duct cleaning ---
        "air duct maintenance cleaning":            "maintenance air duct cleaning",
        "maintenance cleaning":                     "maintenance air duct cleaning",
        "maintenance duct cleaning":                "maintenance air duct cleaning",
        "vacuum maintenance air duct cleaning":     "maintenance air duct cleaning",
        "air duct cleaning":                        "air duct deep cleaning",
        "duct and vent cleaning":                   "air duct deep cleaning",
        "supply vent deep cleaning":                "air duct deep cleaning",
        "return duct deep cleaning":                "air duct deep cleaning",
        "return deep cleaning":                     "air duct deep cleaning",
        "air duct return deep cleaning":            "air duct deep cleaning",

        # --- Blower cleaning ---
        "blower deep cleaning + coil maintenance":  "blower deep cleaning and coil maintenance",
        "blower deep cleaning & coil maintenance":  "blower deep cleaning and coil maintenance",
        "blower deep cleaning":                     "blower cleaning",
        "blower fan cleaning":                      "blower cleaning",
        "blower cleaning & disinfectant":           "blower cleaning",

        # --- UV light ---
        "uv light system + install":                "uv light system and installation",
        "uv light system & installation":           "uv light system and installation",
        "uv light system & installation - plenum":  "uv light system and installation",

        # --- Duct encapsulation ---
        "duct encapsulation - fiberglass":          "duct encapsulation",
        "air duct encapsulation":                   "duct encapsulation",
        "plenum box encapsulation":                 "plenum encapsulation",

        # --- Coil cleaning ---
        "evaporator coil cleaning":                 "coil cleaning",
        "coil drip pan cleaning":                   "coil cleaning",
        "ac unit cleaning":                         "coil cleaning",

        # Dryer vent unclog variants
        "dryer vent cleaning + roof unclogg":       "dryer vent cleaning roof unclog",
        "dryer vent roof unclogg":                  "dryer vent cleaning roof unclog",
        "dryer vent attic unclogg":                 "dryer vent cleaning attic unclog",
        "dryer vent attic unclog":                  "dryer vent cleaning attic unclog",
        "dryer vent unclog from roof":              "dryer vent cleaning roof unclog",
        "dryer vent unclog from wall":              "dryer vent cleaning wall unclog",
        "dryer vent wall unclog":                   "dryer vent cleaning wall unclog",
        "dryer vent - wall unclog":                 "dryer vent cleaning wall unclog",
        "dryer vent - unclog from wall":            "dryer vent cleaning wall unclog",
        "dryer vent cleaning & wall unclog":        "dryer vent cleaning wall unclog",
        "dryer vent cleaning + wall unclog":        "dryer vent cleaning wall unclog",
        "dryer vent cleaning -wall unclog":         "dryer vent cleaning wall unclog",
        "dryer additional length":                  "dryer vent additional length",
        "dryer vent cleaning & additional length":  "dryer vent additional length",

        # Blower variants
        "blower fan deep cleaning":                 "blower cleaning",
        "blower cleaning (in place)":               "blower cleaning",
        "blower fan cleaning - in place":           "blower cleaning",
        "blower fan cleaning in-place":             "blower cleaning",
        "blower fan in-place cleaning":             "blower cleaning",
        "blower deep cleaning & disinfectant":      "blower cleaning",
        "blower cleaning + disinfectant":           "blower cleaning",
        "blower cleaning & disinfect":              "blower cleaning",
        "blower cleaning & disinfection":           "blower cleaning",
        "blower cleaning + disinfecting":           "blower cleaning",
        "blower cleaning (in-place) & disinfectant":"blower cleaning",
        "blower deep cleaning + coil cleaning":     "blower deep cleaning and coil maintenance",
        "blower fan cleaning and coil maintenance": "blower deep cleaning and coil maintenance",
        "blower cleaning + coil maintenance":       "blower deep cleaning and coil maintenance",
        "blower cleaning (in-place) + coil maintenance": "blower deep cleaning and coil maintenance",
        "blower fan & coil cleaning":               "blower deep cleaning and coil maintenance",

        # UV light variants
        "uv light system & installation + 2 y warranty":    "uv light system and installation",
        "uv light system & installation + 1 yw":            "uv light system and installation",
        "uv light system & installation + 1 y warranty":    "uv light system and installation",
        "uv light system & installation + 1 year warranty": "uv light system and installation",
        "uv light system & installation (price match)":     "uv light system and installation",
        "uv light system & installation -plenum":           "uv light system and installation",
        "uv light system & installation - plenum box":      "uv light system and installation",
        "uv light system & installation - plenum (price match)": "uv light system and installation",
        "plenum uv light system & installation":            "uv light system and installation",
        "uv light system for plenum & installation":        "uv light system and installation",
        "uv light system & installation plenum":            "uv light system and installation",
        "hvac  - uv light system & installation":           "uv light system and installation",

        # Duct encapsulation variants
        "duct encapsulation - fibreglass":          "duct encapsulation",
        "return duct encapsulation":                "duct encapsulation",
        "intake duct encapsulation":                "duct encapsulation",
        "planum duct encapsulation":                "plenum encapsulation",
        "duct cleaning":                            "air duct deep cleaning",
        "vacuum air duct cleaning":                 "maintenance air duct cleaning",
        "maintenance vacuum air duct cleaning":     "maintenance air duct cleaning",
    }

    return mappings.get(s, s)


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
    if not tag or not isinstance(tag, str):
        return None
    
    match = re.search(r'[Tt]echnician\s*:\s*([A-Z][A-Z &]+)', tag)
    if not match:
        return None
    
    name = match.group(1).strip()
    # If multiple technicians (e.g. "VIVO&DAVID"), take the first
    name = re.split(r'[&,]', name)[0].strip()
    return name.upper()