"""
upsell_tool — LangChain tool wrapping predict_upsell().

Looks up city_median_job_value from v_city_price_index before calling the model,
so the agent only needs to supply a city name rather than a raw dollar figure.
"""
import datetime
from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from etl.utils import get_connection, normalize_service_name
from ml.models.upsell.predict import predict_upsell


def _city_median(city: str) -> float:
    """Return city median job value from v_city_price_index; fall back to state median."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT median_job_value FROM v_city_price_index WHERE LOWER(city) = LOWER(%s)",
                (city,),
            )
            row = cur.fetchone()
            if row:
                return float(row[0])
            # State median fallback
            cur.execute("SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY job_amount) FROM jobs WHERE job_amount > 0")
            fallback = cur.fetchone()
            return float(fallback[0]) if fallback and fallback[0] else 1350.0


class UpsellInput(BaseModel):
    first_service: str = Field(description="The primary service the customer is booking (e.g. 'air duct deep cleaning')")
    city: str = Field(description="Customer's city (e.g. 'Boca Raton')")
    lead_source: str = Field(default="unknown", description="Lead origin: 'google', 'thumbtack', 'yelp', 'referral', etc.")
    is_returning_customer: bool = Field(default=False, description="True if the customer has booked before")
    job_amount: float = Field(default=0.0, description="Expected job value in dollars (0 if unknown)")
    month: Optional[int] = Field(default=None, description="Month 1-12; defaults to current month if omitted")


@tool("upsell_tool", args_schema=UpsellInput)
def upsell_tool(
    first_service: str,
    city: str,
    lead_source: str = "unknown",
    is_returning_customer: bool = False,
    job_amount: float = 0.0,
    month: Optional[int] = None,
) -> str:
    """
    Predict which additional services a customer is likely to buy in the same job.
    Returns a ranked list of service recommendations with probabilities.
    Use this when a technician wants to know what to pitch on-site, or when the
    owner wants upsell recommendations for an incoming job.
    """
    if month is None:
        month = datetime.date.today().month

    normalized = normalize_service_name(first_service) or first_service
    city_median = _city_median(city)

    job_profile = {
        "first_service": normalized,
        "lead_source": lead_source or "unknown",
        "is_returning_customer": is_returning_customer,
        "prior_job_count": 0,
        "prior_avg_job_value": 0.0,
        "city_median_job_value": city_median,
        "job_amount": job_amount,
        "month": month,
    }

    results = predict_upsell(job_profile)

    lines = [f"Upsell recommendations for '{normalized}' in {city} (city median: ${city_median:,.0f}):"]
    for r in results[:5]:
        flag = "✓ RECOMMEND" if r["recommend"] else "  consider"
        lines.append(
            f"  {flag}  {r['service']:40s}  {r['probability']*100:.0f}%  [{r['confidence']} confidence]"
        )
    return "\n".join(lines)
