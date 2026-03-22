-- ============================================================
-- DuctAI Copilot — Analytical Views
-- Run against a live database: psql $DATABASE_URL -f db/views.sql
-- Safe to re-run: uses CREATE OR REPLACE VIEW
--
-- View index:
--   Business analytics (1-9):
--     1.  v_service_mix
--     2.  v_revenue_summary
--     3.  v_employee_performance
--     4.  v_new_customers_per_month
--     5.  v_region_summary
--     6.  v_lead_source_roi
--     7.  v_estimate_pipeline
--     8.  v_customer_retention
--     9.  v_service_cooccurrence
--   ML support (10-12):
--     10. v_customer_identity    — deduplicates returning customers by phone
--     11. v_customer_history     — per-person job history using canonical ID
--     12. v_city_price_index     — median job value per city for pricing model
-- ============================================================


-- ============================================================
-- BUSINESS ANALYTICS VIEWS
-- ============================================================

-- ------------------------------------------------------------
-- 1. Service mix
-- Q: Which services are most requested and most profitable?
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW v_service_mix AS
SELECT
    li.service_key                              AS service,
    COUNT(DISTINCT li.job_id)                   AS job_count,
    ROUND(SUM(li.price), 2)                     AS total_revenue,
    ROUND(AVG(li.price), 2)                     AS avg_price,
    ROUND(
        COUNT(DISTINCT li.job_id) * 100.0 /
        NULLIF((SELECT COUNT(DISTINCT id) FROM jobs WHERE status = 'Completed'), 0)
    , 1)                                        AS pct_of_completed_jobs
FROM line_items li
JOIN jobs j ON j.id = li.job_id
WHERE j.status = 'Completed'
GROUP BY li.service_key
ORDER BY job_count DESC;


-- ------------------------------------------------------------
-- 2. Revenue summary (monthly trend)
-- Q: What is the company revenue over time?
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW v_revenue_summary AS
SELECT
    DATE_TRUNC('month', completed_at)           AS month,
    COUNT(id)                                   AS job_count,
    ROUND(SUM(job_amount), 2)                   AS total_revenue,
    ROUND(AVG(job_amount), 2)                   AS avg_job_value,
    ROUND(SUM(subtotal), 2)                     AS total_catalog_value,
    ROUND(SUM(discount_amount), 2)              AS total_discounts,
    ROUND(
        SUM(discount_amount) * 100.0 /
        NULLIF(SUM(subtotal), 0)
    , 1)                                        AS discount_rate_pct
FROM jobs
WHERE status = 'Completed'
  AND completed_at IS NOT NULL
GROUP BY DATE_TRUNC('month', completed_at)
ORDER BY month;


-- ------------------------------------------------------------
-- 3. Employee performance
-- Q: Which employee completed the most jobs / highest revenue?
-- Owners always show 0 commission regardless of job amount.
-- Hourly/salary employees show NULL commission (hours not tracked).
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW v_employee_performance AS
SELECT
    e.name                                          AS employee,
    e.role,
    e.pay_type,
    COUNT(j.id)                                     AS job_count,
    ROUND(SUM(j.job_amount), 2)                     AS total_revenue,
    ROUND(AVG(j.job_amount), 2)                     AS avg_job_value,
    ROUND(SUM(j.subtotal), 2)                       AS total_catalog_value,
    ROUND(SUM(j.discount_amount), 2)                AS total_discounts,
    ROUND(
        SUM(j.discount_amount) * 100.0 /
        NULLIF(SUM(j.subtotal), 0)
    , 1)                                            AS discount_rate_pct,
    ROUND(
        CASE e.role
            WHEN 'owner' THEN 0
            ELSE
                CASE e.pay_type
                    WHEN 'commission' THEN
                        SUM(j.job_amount) * COALESCE(e.commission_rate, 0)
                    WHEN 'tiered' THEN
                        SUM(
                            CASE
                                WHEN j.job_amount >= e.commission_tier2_threshold
                                    THEN j.job_amount * e.commission_tier2_rate
                                WHEN j.job_amount >= e.commission_tier1_threshold
                                    THEN j.job_amount * e.commission_tier1_rate
                                ELSE
                                    j.job_amount * COALESCE(e.commission_rate, 0)
                            END
                        )
                    ELSE NULL
                END
        END
    , 2)                                            AS estimated_commission
FROM employees e
JOIN jobs j ON j.employee_id = e.id
WHERE j.status = 'Completed'
GROUP BY e.name, e.role, e.pay_type, e.commission_rate,
         e.commission_tier1_rate, e.commission_tier1_threshold,
         e.commission_tier2_rate, e.commission_tier2_threshold
ORDER BY total_revenue DESC;


-- ------------------------------------------------------------
-- 4. New customers per month
-- Q: How many new customers this month vs previous month?
-- Note: counts customer records not unique people.
-- See v_customer_identity for true person deduplication.
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW v_new_customers_per_month AS
SELECT
    DATE_TRUNC('month', created_at)             AS month,
    COUNT(id)                                   AS new_customers,
    SUM(COUNT(id)) OVER (
        ORDER BY DATE_TRUNC('month', created_at)
    )                                           AS cumulative_customers
FROM customers
WHERE created_at IS NOT NULL
GROUP BY DATE_TRUNC('month', created_at)
ORDER BY month;


-- ------------------------------------------------------------
-- 5. Region summary
-- Q: Which regions have the most customers and highest revenue?
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW v_region_summary AS
SELECT
    c.state,
    c.city,
    c.zip,
    COUNT(DISTINCT c.id)                        AS customer_count,
    COUNT(j.id)                                 AS job_count,
    ROUND(SUM(j.job_amount), 2)                 AS total_revenue,
    ROUND(AVG(j.job_amount), 2)                 AS avg_job_value
FROM customers c
LEFT JOIN jobs j ON j.customer_id = c.id
    AND j.status = 'Completed'
GROUP BY c.state, c.city, c.zip
ORDER BY total_revenue DESC NULLS LAST;


-- ------------------------------------------------------------
-- 6. Lead source ROI
-- Q: Which lead source brings the most valuable customers?
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW v_lead_source_roi AS
SELECT
    COALESCE(c.lead_source, 'unknown')          AS lead_source,
    COUNT(DISTINCT c.id)                        AS customer_count,
    COUNT(j.id)                                 AS job_count,
    ROUND(SUM(j.job_amount), 2)                 AS total_revenue,
    ROUND(AVG(j.job_amount), 2)                 AS avg_job_value,
    ROUND(
        COUNT(j.id) * 1.0 /
        NULLIF(COUNT(DISTINCT c.id), 0)
    , 2)                                        AS jobs_per_customer,
    ROUND(
        SUM(j.job_amount) /
        NULLIF(COUNT(DISTINCT c.id), 0)
    , 2)                                        AS revenue_per_customer
FROM customers c
LEFT JOIN jobs j ON j.customer_id = c.id
    AND j.status = 'Completed'
GROUP BY COALESCE(c.lead_source, 'unknown')
ORDER BY total_revenue DESC NULLS LAST;


-- ------------------------------------------------------------
-- 7. Estimate pipeline
-- Q: What is the open revenue pipeline and conversion rate?
-- Uses consolidated `value` column (won/lost/open merged in ETL).
-- Two conversion rates — see data_notes.md for explanation.
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW v_estimate_pipeline AS
SELECT
    COUNT(*)                                                AS total_estimates,
    COUNT(*) FILTER (WHERE outcome = 'won')                 AS won_count,
    COUNT(*) FILTER (WHERE outcome = 'lost')                AS lost_count,
    COUNT(*) FILTER (WHERE outcome = 'open')                AS open_count,
    -- Decision conversion: won / (won + lost) — excludes still-open
    ROUND(
        COUNT(*) FILTER (WHERE outcome = 'won') * 100.0 /
        NULLIF(COUNT(*) FILTER (WHERE outcome IN ('won','lost')), 0)
    , 1)                                                    AS decision_conversion_pct,
    -- Overall conversion: won / all estimates including open
    ROUND(
        COUNT(*) FILTER (WHERE outcome = 'won') * 100.0 /
        NULLIF(COUNT(*), 0)
    , 1)                                                    AS overall_conversion_pct,
    ROUND(SUM(value) FILTER (WHERE outcome = 'won'), 2)     AS total_won_value,
    ROUND(SUM(value) FILTER (WHERE outcome = 'open'), 2)    AS total_open_pipeline,
    ROUND(AVG(value) FILTER (WHERE outcome = 'won'), 2)     AS avg_won_value,
    ROUND(AVG(value) FILTER (WHERE outcome = 'lost'), 2)    AS avg_lost_value
FROM estimates;


-- ------------------------------------------------------------
-- 8. Customer retention segments
-- Q: What share of revenue comes from repeat customers?
-- Note: uses customer_id directly — may undercount true returning
-- customers due to Jobber duplicate records. See v_customer_identity.
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW v_customer_retention AS
SELECT
    CASE
        WHEN job_count = 1 THEN 'one-time'
        WHEN job_count BETWEEN 2 AND 3 THEN 'returning (2-3 jobs)'
        ELSE 'loyal (4+ jobs)'
    END                                         AS customer_segment,
    COUNT(*)                                    AS customer_count,
    ROUND(SUM(total_revenue), 2)                AS total_revenue,
    ROUND(AVG(avg_job_value), 2)                AS avg_job_value,
    ROUND(AVG(job_count), 1)                    AS avg_jobs
FROM (
    SELECT
        c.id,
        COUNT(j.id)                             AS job_count,
        SUM(j.job_amount)                       AS total_revenue,
        AVG(j.job_amount)                       AS avg_job_value
    FROM customers c
    LEFT JOIN jobs j ON j.customer_id = c.id
        AND j.status = 'Completed'
    GROUP BY c.id
) customer_summary
GROUP BY customer_segment
ORDER BY customer_count DESC;


-- ------------------------------------------------------------
-- 9. Service co-occurrence (ML foundation)
-- Q: Which services are purchased together in the same job?
-- Directly feeds the upsell model in Phase 1.
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW v_service_cooccurrence AS
SELECT
    a.service_key                               AS service_a,
    b.service_key                               AS service_b,
    COUNT(*)                                    AS co_count,
    ROUND(
        COUNT(*) * 100.0 /
        NULLIF((SELECT COUNT(DISTINCT job_id) FROM line_items), 0)
    , 2)                                        AS pct_of_jobs
FROM line_items a
JOIN line_items b
    ON  a.job_id = b.job_id
    AND a.service_key < b.service_key
GROUP BY a.service_key, b.service_key
ORDER BY co_count DESC;


-- ============================================================
-- ML SUPPORT VIEWS
-- ============================================================

-- ------------------------------------------------------------
-- 10. Customer identity deduplication
-- Problem: Jobber creates a new customer_id for returning
-- customers booked via pipeline automation or Thumbtack imports.
-- 81 phone numbers are linked to 2+ customer IDs.
-- 47 are confirmed returning customers with new records.
-- 21 are double-entry errors (created within minutes of each other).
--
-- Solution: canonical_id = earliest ID sharing the same phone.
-- This is the true person identifier for all ML features.
--
-- Usage: always JOIN through this view before aggregating
-- job history for ML feature building.
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW v_customer_identity AS
SELECT
    c.id                                        AS customer_id,
    CASE
        WHEN c.phone IS NOT NULL THEN
            FIRST_VALUE(c.id) OVER (
                PARTITION BY c.phone
                ORDER BY c.created_at ASC
                ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
            )
        ELSE c.id
    END                                         AS canonical_id,
    c.phone,
    c.name                                      AS display_name,
    c.lead_source,
    c.city,
    c.state,
    c.zip,
    c.created_at
FROM customers c
WHERE c.name NOT ILIKE '%express job%'
  AND c.name NOT ILIKE '%external job%'
  AND c.name NOT ILIKE '%reminder%';


-- ------------------------------------------------------------
-- 11. Customer history (deduplication-aware)
-- Aggregates all jobs per canonical person, not per record ID.
-- Correctly counts job history for returning customers who
-- were re-created in Jobber with a new customer ID.
--
-- ML features:
--   prior_job_count       — total completed jobs as a person
--   prior_avg_job_value   — average spend across all their jobs
--   prior_max_job_value   — highest single job (upsell ceiling)
--   is_returning_customer — has at least one prior completed job
--   last_job_date         — recency signal
--   lead_source           — fills the 97% null gap in estimates
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW v_customer_history AS
SELECT
    ci.canonical_id,
    MAX(ci.lead_source)                         AS lead_source,
    COUNT(j.id)                                 AS prior_job_count,
    ROUND(AVG(j.job_amount), 2)                 AS prior_avg_job_value,
    ROUND(MAX(j.job_amount), 2)                 AS prior_max_job_value,
    MAX(j.completed_at)                         AS last_job_date,
    CASE
        WHEN COUNT(j.id) > 0 THEN TRUE
        ELSE FALSE
    END                                         AS is_returning_customer
FROM v_customer_identity ci
LEFT JOIN jobs j ON j.customer_id = ci.customer_id
    AND j.status = 'Completed'
    AND j.job_amount > 0
GROUP BY ci.canonical_id;


-- ------------------------------------------------------------
-- 12. City price index
-- Pre-computes median and percentile job values per city.
-- Used by the pricing model as a market price signal.
-- Cities with fewer than 3 completed jobs are excluded —
-- insufficient data for a reliable median.
-- In the ML pipeline, low-sample cities fall back to state median.
--
-- ML features:
--   median_job_value — central price for this market
--   p25_job_value    — lower bound of typical range
--   p75_job_value    — upper bound of typical range
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW v_city_price_index AS
SELECT
    city,
    state,
    COUNT(id)                                   AS job_count,
    ROUND(
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY job_amount)
    , 2)                                        AS median_job_value,
    ROUND(AVG(job_amount), 2)                   AS avg_job_value,
    ROUND(
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY job_amount)
    , 2)                                        AS p25_job_value,
    ROUND(
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY job_amount)
    , 2)                                        AS p75_job_value
FROM jobs
WHERE status = 'Completed'
  AND job_amount > 0
  AND city IS NOT NULL
GROUP BY city, state
HAVING COUNT(id) >= 3
ORDER BY median_job_value DESC;