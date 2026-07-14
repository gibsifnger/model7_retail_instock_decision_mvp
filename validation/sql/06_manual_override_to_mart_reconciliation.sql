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

centers(center_id) AS (
    VALUES
        ('MFC_GANGNAM'),
        ('FC_DONGTAN'),
        ('FC_INCHEON')
),

base_grid AS (
    SELECT
        s.sku_id,
        c.channel_id,
        f.center_id,
        w.week_start_date
    FROM sku_master AS s
    CROSS JOIN channels AS c
    CROSS JOIN centers AS f
    CROSS JOIN weeks AS w
),

override_candidates AS (
    SELECT
        g.sku_id,
        g.channel_id,
        g.center_id,
        g.week_start_date,

        o.override_id,
        o.override_type,
        o.override_reason,
        o.approval_status,
        o.created_at,

        COUNT(*) OVER (
            PARTITION BY
                g.sku_id,
                g.channel_id,
                g.center_id,
                g.week_start_date
        ) AS candidate_count,

        ROW_NUMBER() OVER (
            PARTITION BY
                g.sku_id,
                g.channel_id,
                g.center_id,
                g.week_start_date
            ORDER BY
                datetime(o.created_at) DESC,
                o.override_id
        ) AS override_rank

    FROM base_grid AS g

    INNER JOIN manual_overrides AS o
        ON o.sku_id = g.sku_id
       AND (
            NULLIF(o.channel_id, '') IS NULL
            OR o.channel_id = g.channel_id
       )
       AND (
            NULLIF(o.center_id, '') IS NULL
            OR o.center_id = g.center_id
       )
       AND g.week_start_date
            BETWEEN date(o.effective_from) AND date(o.effective_to)
       AND o.approval_status = 'APPROVED'
),

expected_override AS (
    SELECT
        sku_id,
        channel_id,
        center_id,
        week_start_date,
        override_type,
        override_reason,
        candidate_count
    FROM override_candidates
    WHERE override_rank = 1
),

comparison AS (
    SELECT
        g.sku_id,
        g.channel_id,
        g.center_id,
        g.week_start_date,

        CASE
            WHEN e.sku_id IS NULL THEN 0
            ELSE 1
        END AS expected_override_flag,

        m.manual_override_flag AS mart_override_flag,

        CASE
            WHEN
                CASE WHEN e.sku_id IS NULL THEN 0 ELSE 1 END
                <> COALESCE(m.manual_override_flag, 0)
            THEN 1
            ELSE 0
        END AS override_flag_mismatch,

        CASE
            WHEN e.sku_id IS NOT NULL
             AND (
                    COALESCE(e.override_type, '')
                        <> COALESCE(m.override_type, '')
                 OR COALESCE(e.override_reason, '')
                        <> COALESCE(m.override_reason, '')
             )
            THEN 1
            ELSE 0
        END AS override_value_mismatch

    FROM base_grid AS g

    LEFT JOIN expected_override AS e
        ON e.sku_id = g.sku_id
       AND e.channel_id = g.channel_id
       AND e.center_id = g.center_id
       AND e.week_start_date = g.week_start_date

    INNER JOIN mart_retail_instock_weekly AS m
        ON m.sku_id = g.sku_id
       AND m.channel_id = g.channel_id
       AND m.center_id = g.center_id
       AND m.week_start_date = g.week_start_date
)

SELECT
    (SELECT COUNT(*)
     FROM manual_overrides)
        AS raw_override_rows,

    (SELECT COUNT(*)
     FROM manual_overrides
     WHERE approval_status = 'APPROVED')
        AS approved_override_rows,

    (SELECT COUNT(*)
     FROM manual_overrides
     WHERE approval_status <> 'APPROVED')
        AS non_approved_override_rows,

    (SELECT COUNT(*)
     FROM override_candidates)
        AS eligible_expanded_candidate_rows,

    (SELECT COUNT(*)
     FROM expected_override)
        AS expected_selected_override_keys,

    (SELECT COUNT(*)
     FROM expected_override
     WHERE candidate_count > 1)
        AS overlapping_override_keys,

    (SELECT COUNT(*)
     FROM comparison
     WHERE override_flag_mismatch = 1)
        AS override_flag_mismatch_count,

    (SELECT COUNT(*)
     FROM comparison
     WHERE override_value_mismatch = 1)
        AS override_value_mismatch_count,

    (SELECT COUNT(*)
     FROM mart_retail_instock_weekly
     WHERE manual_override_flag = 1)
        AS mart_selected_override_keys,

    (SELECT COUNT(*)
     FROM mart_retail_instock_weekly)
        AS mart_total_rows;
