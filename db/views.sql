-- ============================================================
-- DuctAI Copilot — Analytical Views
-- Run against a live database: psql $DATABASE_URL -f db/views.sql
-- Safe to re-run: uses CREATE OR REPLACE VIEW
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
-- Hourly/salary employees show NULL commission (not calculable
-- from job data alone — hours not tracked in CRM).
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
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW v_estimate_pipeline AS
SELECT
    COUNT(*)                                                        AS total_estimates,
    COUNT(*) FILTER (WHERE outcome = 'won')                         AS won_count,
    COUNT(*) FILTER (WHERE outcome = 'lost')                        AS lost_count,
    COUNT(*) FILTER (WHERE outcome = 'open')                        AS open_count,
    ROUND(
        COUNT(*) FILTER (WHERE outcome = 'won') * 100.0 /
        NULLIF(COUNT(*) FILTER (WHERE outcome IN ('won','lost')), 0)
    , 1)                                                            AS conversion_rate_pct,
    ROUND(SUM(won_value), 2)                                        AS total_won_value,
    ROUND(SUM(open_value), 2)                                       AS total_open_pipeline,
    ROUND(AVG(won_value) FILTER (WHERE outcome = 'won'), 2)         AS avg_won_value
FROM estimates;


-- ------------------------------------------------------------
-- 8. Customer retention segments
-- Q: What share of revenue comes from repeat customers?
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