WITH RECURSIVE
weeks(week_start_date) AS (
    SELECT date('2025-01-06')

    UNION ALL

    SELECT date(week_start_date, '+7 days')
    FROM weeks
    WHERE week_start_date < date('2025-12-29')
),

channels(channel_id) AS (
    VALUES
        ('ONLINE_MALL'),
        ('ROCKET_DELIVERY'),
        ('GLOBAL_MALL')
),

sku_channel_week AS (
    SELECT
        s.sku_id,
        c.channel_id,
        w.week_start_date
    FROM sku_master AS s
    CROSS JOIN channels AS c
    CROSS JOIN weeks AS w
),

promotion_candidates AS (
    SELECT
        g.sku_id,
        g.channel_id,
        g.week_start_date,

        p.promotion_id,
        p.promo_type,
        p.planned_discount_rate AS promo_depth,
        p.planned_promo_price,
        p.promo_priority,

        ROW_NUMBER() OVER (
            PARTITION BY
                g.sku_id,
                g.channel_id,
                g.week_start_date
            ORDER BY
                CASE p.promo_priority
                    WHEN 'HIGH' THEN 1
                    WHEN 'MEDIUM' THEN 2
                    WHEN 'LOW' THEN 3
                    ELSE 4
                END,
                p.planned_discount_rate DESC,
                p.promotion_id
        ) AS promo_rank,

        COUNT(*) OVER (
            PARTITION BY
                g.sku_id,
                g.channel_id,
                g.week_start_date
        ) AS candidate_count

    FROM sku_channel_week AS g

    INNER JOIN sku_code_mapping AS m
        ON m.sku_id = g.sku_id
       AND m.source_system = 'MD'
       AND m.mapping_status = 'VALID'

    INNER JOIN md_promotion_calendar AS p
        ON p.md_sku_code = m.source_sku_code
       AND p.channel_id = g.channel_id
       AND date(p.promo_start_date)
            <= date(g.week_start_date, '+6 days')
       AND date(p.promo_end_date)
            >= g.week_start_date
       AND date(p.promo_start_date)
            >= date(m.effective_from)
       AND (
            NULLIF(m.effective_to, '') IS NULL
            OR date(p.promo_end_date)
               <= date(m.effective_to)
       )
),

expected_promotion AS (
    SELECT
        sku_id,
        channel_id,
        week_start_date,
        promo_type,
        promo_depth,
        planned_promo_price,
        promo_priority,
        candidate_count
    FROM promotion_candidates
    WHERE promo_rank = 1
),

mart_promotion AS (
    SELECT
        sku_id,
        channel_id,
        week_start_date,

        COUNT(*) AS center_row_count,

        MIN(promo_flag) AS min_promo_flag,
        MAX(promo_flag) AS max_promo_flag,

        MIN(promo_type) AS min_promo_type,
        MAX(promo_type) AS max_promo_type,

        MIN(promo_depth) AS min_promo_depth,
        MAX(promo_depth) AS max_promo_depth,

        MIN(planned_promo_price) AS min_promo_price,
        MAX(planned_promo_price) AS max_promo_price,

        MIN(promo_priority) AS min_promo_priority,
        MAX(promo_priority) AS max_promo_priority

    FROM mart_retail_instock_weekly

    GROUP BY
        sku_id,
        channel_id,
        week_start_date
),

comparison AS (
    SELECT
        g.sku_id,
        g.channel_id,
        g.week_start_date,

        CASE
            WHEN e.sku_id IS NULL THEN 0
            ELSE 1
        END AS expected_promo_flag,

        m.min_promo_flag AS mart_promo_flag,

        CASE
            WHEN COALESCE(
                    CASE WHEN e.sku_id IS NULL THEN 0 ELSE 1 END,
                    0
                 ) <> COALESCE(m.min_promo_flag, 0)
            THEN 1
            ELSE 0
        END AS promo_flag_mismatch,

        CASE
            WHEN e.sku_id IS NOT NULL
             AND (
                    COALESCE(e.promo_type, '')
                        <> COALESCE(m.min_promo_type, '')
                 OR COALESCE(e.promo_depth, -1)
                        <> COALESCE(m.min_promo_depth, -1)
                 OR COALESCE(e.planned_promo_price, -1)
                        <> COALESCE(m.min_promo_price, -1)
                 OR COALESCE(e.promo_priority, '')
                        <> COALESCE(m.min_promo_priority, '')
             )
            THEN 1
            ELSE 0
        END AS promo_value_mismatch

    FROM sku_channel_week AS g

    LEFT JOIN expected_promotion AS e
        ON e.sku_id = g.sku_id
       AND e.channel_id = g.channel_id
       AND e.week_start_date = g.week_start_date

    LEFT JOIN mart_promotion AS m
        ON m.sku_id = g.sku_id
       AND m.channel_id = g.channel_id
       AND m.week_start_date = g.week_start_date
)

SELECT
    (SELECT COUNT(*)
     FROM md_promotion_calendar)
        AS raw_promotion_rows,

    (SELECT COUNT(*)
     FROM promotion_candidates)
        AS eligible_promotion_candidate_rows,

    (SELECT COUNT(*)
     FROM expected_promotion)
        AS expected_selected_promotion_keys,

    (SELECT COUNT(*)
     FROM expected_promotion
     WHERE candidate_count > 1)
        AS overlapping_promotion_keys,

    (SELECT COUNT(*)
     FROM mart_promotion)
        AS mart_sku_channel_week_keys,

    (SELECT COUNT(*)
     FROM mart_promotion
     WHERE center_row_count <> 3)
        AS unexpected_center_row_count,

    (SELECT COUNT(*)
     FROM mart_promotion
     WHERE
        min_promo_flag <> max_promo_flag
        OR COALESCE(min_promo_type, '')
            <> COALESCE(max_promo_type, '')
        OR COALESCE(min_promo_depth, -1)
            <> COALESCE(max_promo_depth, -1)
        OR COALESCE(min_promo_price, -1)
            <> COALESCE(max_promo_price, -1)
        OR COALESCE(min_promo_priority, '')
            <> COALESCE(max_promo_priority, ''))
        AS center_repeated_value_mismatch_count,

    (SELECT COUNT(*)
     FROM comparison
     WHERE promo_flag_mismatch = 1)
        AS promo_flag_mismatch_count,

    (SELECT COUNT(*)
     FROM comparison
     WHERE promo_value_mismatch = 1)
        AS promo_value_mismatch_count,

    (SELECT COUNT(*)
     FROM mart_promotion
     WHERE min_promo_flag = 1)
        AS mart_selected_promotion_keys;
