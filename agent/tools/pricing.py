"""
pricing_tool — LangChain tool wrapping predict_price().

Looks up city_median_job_value and city_rejection_rate from v_city_price_index,
derives season from month, then calls the quantile regression model.
"""
import datetime
from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from etl.utils import get_connection
from ml.models.pricing.predict import predict_price


def _month_to_season(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "fall"


def _city_stats(city: str) -> tuple[float, float]:
    """Return (median_job_value, city_rejection_rate) for city; falls back to state median.

    v_city_price_index has no rejection_rate column — we derive a proxy from
    the ratio of estimates lost to total estimates in that city.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT median_job_value FROM v_city_price_index WHERE LOWER(city) = LOWER(%s)",
                (city,),
            )
            row = cur.fetchone()
            if row:
                city_median = float(row[0])
            else:
                # State median fallback
                cur.execute(
                    "SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY job_amount) FROM jobs WHERE job_amount > 0"
                )
                fallback = cur.fetchone()
                city_median = float(fallback[0]) if fallback and fallback[0] else 1350.0

            # Derive rejection rate: fraction of lost estimates in this city
            cur.execute(
                """
                SELECT
                    COUNT(*) FILTER (WHERE outcome = 'lost') * 1.0 / NULLIF(COUNT(*), 0)
                FROM estimates e
                JOIN customers c ON c.id = e.customer_id
                WHERE LOWER(c.city) = LOWER(%s)
                """,
                (city,),
            )
            rate_row = cur.fetchone()
            rejection_rate = float(rate_row[0]) if rate_row and rate_row[0] is not None else 0.35

    return city_median, rejection_rate


class PricingInput(BaseModel):
    city: str = Field(description="Customer's city (e.g. 'Boca Raton')")
    lead_source: str = Field(default="unknown", description="Lead origin: 'google', 'thumbtack', 'yelp', 'referral', etc.")
    is_returning_customer: bool = Field(default=False, description="True if customer has booked before")
    prior_avg_job_value: float = Field(default=0.0, description="Average dollar value of prior jobs for this customer (0 if new)")
    month: Optional[int] = Field(default=None, description="Month 1-12; defaults to current month if omitted")


@tool("pricing_tool", args_schema=PricingInput)
def pricing_tool(
    city: str,
    lead_source: str = "unknown",
    is_returning_customer: bool = False,
    prior_avg_job_value: float = 0.0,
    month: Optional[int] = None,
) -> str:
    """
    Predict a price range for an incoming job estimate.
    Returns a P25–P75 range, median, confidence tier, and a limitation note.
    Use this when the owner wants to know what to quote for a new lead.
    NOTE: This is an internal rough band only — do NOT share these numbers directly with customers.
    """
    if month is None:
        month = datetime.date.today().month

    city_median, city_rejection_rate = _city_stats(city)
    season = _month_to_season(month)

    job_profile = {
        "lead_source": lead_source or "unknown",
        "is_returning_customer": is_returning_customer,
        "prior_avg_job_value": prior_avg_job_value,
        "prior_job_count": 0,
        "city_median_job_value": city_median,
        "city_rejection_rate": city_rejection_rate,
        "month": month,
        "season": season,
    }

    result = predict_price(job_profile)
    low, high = result["range"]

    lines = [
        f"Price estimate for {city} ({lead_source}, {'returning' if is_returning_customer else 'new'} customer):",
        f"  Range:      ${low:,.0f} – ${high:,.0f}",
        f"  Median:     ${result['median']:,.0f}",
        f"  Confidence: {result['confidence']} ({result['n_similar']} similar past jobs)",
        f"  City median: ${city_median:,.0f} | Season: {season}",
        "",
        "  ⚠ INTERNAL USE ONLY — rough band for owner guidance, not for customer-facing quotes.",
        "  The P50 median has weak predictive power (limited by training data). Use the range as a sanity check.",
    ]
    return "\n".join(lines)
