WITH RECURSIVE
weeks(week_start_date) AS (
    SELECT date('2025-01-06')

    UNION ALL

    SELECT date(week_start_date, '+7 days')
    FROM weeks
    WHERE week_start_date < date('2025-12-29')
),

centers(center_id) AS (
    VALUES
        ('MFC_GANGNAM'),
        ('FC_DONGTAN'),
        ('FC_INCHEON')
),

sku_center_week AS (
    SELECT
        s.sku_id,
        c.center_id,
        w.week_start_date
    FROM sku_master AS s
    CROSS JOIN centers AS c
    CROSS JOIN weeks AS w
),

po_mapped AS (
    SELECT
        m.sku_id,
        p.po_id,
        p.po_line_id,
        date(p.po_created_date) AS po_created_date,
        p.center_id,
        p.confirmed_qty,
        p.cancelled_qty,
        date(p.promised_delivery_date) AS promised_delivery_date

    FROM erp_purchase_orders AS p

    INNER JOIN sku_code_mapping AS m
        ON m.source_system = 'ERP'
       AND m.source_sku_code = p.erp_sku_code
       AND m.mapping_status = 'VALID'
       AND date(p.po_created_date) >= date(m.effective_from)
       AND (
            NULLIF(m.effective_to, '') IS NULL
            OR date(p.po_created_date) <= date(m.effective_to)
       )
),

receipt_mapped AS (
    SELECT
        m.sku_id,
        r.po_id,
        r.po_line_id,
        r.center_id,
        datetime(r.receipt_datetime) AS receipt_datetime,
        r.received_qty,
        r.accepted_qty,
        r.rejected_qty

    FROM wms_goods_receipts AS r

    INNER JOIN sku_code_mapping AS m
        ON m.source_system = 'WMS'
       AND m.source_sku_code = r.wms_sku_code
       AND m.mapping_status = 'VALID'
       AND date(r.receipt_datetime) >= date(m.effective_from)
       AND (
            NULLIF(m.effective_to, '') IS NULL
            OR date(r.receipt_datetime) <= date(m.effective_to)
       )
),

receipt_link_check AS (
    SELECT
        r.*,

        CASE
            WHEN p.po_id IS NULL THEN 1
            ELSE 0
        END AS unmatched_po_flag,

        CASE
            WHEN p.po_id IS NOT NULL
             AND datetime(r.receipt_datetime) < datetime(p.po_created_date)
            THEN 1
            ELSE 0
        END AS receipt_before_po_flag

    FROM receipt_mapped AS r

    LEFT JOIN po_mapped AS p
        ON p.po_id = r.po_id
       AND p.po_line_id = r.po_line_id
       AND p.sku_id = r.sku_id
       AND p.center_id = r.center_id
),

po_lifetime_receipts AS (
    SELECT
        p.sku_id,
        p.center_id,
        p.po_id,
        p.po_line_id,
        p.confirmed_qty,
        COALESCE(SUM(r.accepted_qty), 0) AS total_accepted_qty

    FROM po_mapped AS p

    LEFT JOIN receipt_mapped AS r
        ON r.po_id = p.po_id
       AND r.po_line_id = p.po_line_id
       AND r.sku_id = p.sku_id
       AND r.center_id = p.center_id

    GROUP BY
        p.sku_id,
        p.center_id,
        p.po_id,
        p.po_line_id,
        p.confirmed_qty
),

po_position_by_week AS (
    SELECT
        g.sku_id,
        g.center_id,
        g.week_start_date,

        p.po_id,
        p.po_line_id,
        p.promised_delivery_date,
        p.confirmed_qty,

        COALESCE(
            SUM(
                CASE
                    WHEN r.receipt_datetime
                         < datetime(g.week_start_date, '+8 hours')
                    THEN r.accepted_qty
                    ELSE 0
                END
            ),
            0
        ) AS accepted_qty_as_of_week

    FROM sku_center_week AS g

    INNER JOIN po_mapped AS p
        ON p.sku_id = g.sku_id
       AND p.center_id = g.center_id
       AND p.po_created_date <= g.week_start_date

    LEFT JOIN receipt_mapped AS r
        ON r.po_id = p.po_id
       AND r.po_line_id = p.po_line_id
       AND r.sku_id = p.sku_id
       AND r.center_id = p.center_id

    GROUP BY
        g.sku_id,
        g.center_id,
        g.week_start_date,
        p.po_id,
        p.po_line_id,
        p.promised_delivery_date,
        p.confirmed_qty
),

po_remaining_by_week AS (
    SELECT
        sku_id,
        center_id,
        week_start_date,
        promised_delivery_date,

        CASE
            WHEN confirmed_qty - accepted_qty_as_of_week > 0
            THEN confirmed_qty - accepted_qty_as_of_week
            ELSE 0
        END AS remaining_qty

    FROM po_position_by_week
),

expected_po_nonzero AS (
    SELECT
        sku_id,
        center_id,
        week_start_date,

        SUM(
            CASE
                WHEN promised_delivery_date >= week_start_date
                 AND promised_delivery_date < date(week_start_date, '+7 days')
                THEN remaining_qty
                ELSE 0
            END
        ) AS inbound_qty_next_1w,

        SUM(
            CASE
                WHEN promised_delivery_date >= week_start_date
                 AND promised_delivery_date < date(week_start_date, '+14 days')
                THEN remaining_qty
                ELSE 0
            END
        ) AS inbound_qty_next_2w,

        SUM(
            CASE
                WHEN promised_delivery_date >= week_start_date
                 AND promised_delivery_date < date(week_start_date, '+28 days')
                THEN remaining_qty
                ELSE 0
            END
        ) AS inbound_qty_next_4w,

        SUM(remaining_qty) AS open_po_qty,

        SUM(
            CASE
                WHEN promised_delivery_date < week_start_date
                THEN remaining_qty
                ELSE 0
            END
        ) AS overdue_po_qty

    FROM po_remaining_by_week

    GROUP BY
        sku_id,
        center_id,
        week_start_date
),

expected_po_weekly AS (
    SELECT
        g.sku_id,
        g.center_id,
        g.week_start_date,

        COALESCE(p.inbound_qty_next_1w, 0)
            AS inbound_qty_next_1w,

        COALESCE(p.inbound_qty_next_2w, 0)
            AS inbound_qty_next_2w,

        COALESCE(p.inbound_qty_next_4w, 0)
            AS inbound_qty_next_4w,

        COALESCE(p.open_po_qty, 0)
            AS open_po_qty,

        COALESCE(p.overdue_po_qty, 0)
            AS overdue_po_qty

    FROM sku_center_week AS g

    LEFT JOIN expected_po_nonzero AS p
        ON p.sku_id = g.sku_id
       AND p.center_id = g.center_id
       AND p.week_start_date = g.week_start_date
),

mart_center_week AS (
    SELECT
        sku_id,
        center_id,
        week_start_date,

        COUNT(*) AS channel_row_count,

        MIN(inbound_qty_next_1w) AS min_inbound_qty_next_1w,
        MAX(inbound_qty_next_1w) AS max_inbound_qty_next_1w,

        MIN(inbound_qty_next_2w) AS min_inbound_qty_next_2w,
        MAX(inbound_qty_next_2w) AS max_inbound_qty_next_2w,

        MIN(inbound_qty_next_4w) AS min_inbound_qty_next_4w,
        MAX(inbound_qty_next_4w) AS max_inbound_qty_next_4w,

        MIN(open_po_qty) AS min_open_po_qty,
        MAX(open_po_qty) AS max_open_po_qty,

        MIN(overdue_po_qty) AS min_overdue_po_qty,
        MAX(overdue_po_qty) AS max_overdue_po_qty

    FROM mart_retail_instock_weekly

    GROUP BY
        sku_id,
        center_id,
        week_start_date
),

row_comparison AS (
    SELECT
        e.sku_id,
        e.center_id,
        e.week_start_date,

        CASE
            WHEN m.sku_id IS NULL THEN 1
            ELSE 0
        END AS missing_in_mart,

        CASE
            WHEN
                e.inbound_qty_next_1w <> m.min_inbound_qty_next_1w
                OR e.inbound_qty_next_2w <> m.min_inbound_qty_next_2w
                OR e.inbound_qty_next_4w <> m.min_inbound_qty_next_4w
                OR e.open_po_qty <> m.min_open_po_qty
                OR e.overdue_po_qty <> m.min_overdue_po_qty
            THEN 1
            ELSE 0
        END AS value_mismatch

    FROM expected_po_weekly AS e

    LEFT JOIN mart_center_week AS m
        ON m.sku_id = e.sku_id
       AND m.center_id = e.center_id
       AND m.week_start_date = e.week_start_date
)

SELECT
    (SELECT COUNT(*) FROM erp_purchase_orders)
        AS raw_po_rows,

    (SELECT COUNT(*) FROM po_mapped)
        AS mapped_po_rows,

    (SELECT COUNT(*) FROM wms_goods_receipts)
        AS raw_receipt_rows,

    (SELECT COUNT(*) FROM receipt_mapped)
        AS mapped_receipt_rows,

    (SELECT COUNT(*)
     FROM receipt_link_check
     WHERE unmatched_po_flag = 1)
        AS unmatched_receipt_rows,

    (SELECT COUNT(*)
     FROM receipt_link_check
     WHERE receipt_before_po_flag = 1)
        AS receipt_before_po_rows,

    (SELECT COUNT(*)
     FROM po_lifetime_receipts
     WHERE total_accepted_qty > confirmed_qty)
        AS accepted_exceeds_confirmed_po_lines,

    (SELECT COUNT(*) FROM expected_po_weekly)
        AS expected_center_week_keys,

    (SELECT COUNT(*) FROM mart_center_week)
        AS mart_center_week_keys,

    (SELECT COUNT(*)
     FROM mart_center_week
     WHERE channel_row_count <> 3)
        AS unexpected_channel_row_count,

    (SELECT COUNT(*)
     FROM mart_center_week
     WHERE
        min_inbound_qty_next_1w <> max_inbound_qty_next_1w
        OR min_inbound_qty_next_2w <> max_inbound_qty_next_2w
        OR min_inbound_qty_next_4w <> max_inbound_qty_next_4w
        OR min_open_po_qty <> max_open_po_qty
        OR min_overdue_po_qty <> max_overdue_po_qty)
        AS channel_repeated_value_mismatch_count,

    (SELECT COUNT(*)
     FROM row_comparison
     WHERE missing_in_mart = 1)
        AS expected_keys_missing_in_mart,

    (SELECT COUNT(*)
     FROM row_comparison
     WHERE value_mismatch = 1)
        AS raw_to_mart_value_mismatch_count,

    (SELECT SUM(open_po_qty)
     FROM expected_po_weekly)
        AS expected_open_po_qty_sum,

    (SELECT SUM(min_open_po_qty)
     FROM mart_center_week)
        AS mart_open_po_qty_sum,

    (SELECT SUM(inbound_qty_next_1w)
     FROM expected_po_weekly)
        AS expected_inbound_1w_sum,

    (SELECT SUM(min_inbound_qty_next_1w)
     FROM mart_center_week)
        AS mart_inbound_1w_sum,

    (SELECT SUM(inbound_qty_next_2w)
     FROM expected_po_weekly)
        AS expected_inbound_2w_sum,

    (SELECT SUM(min_inbound_qty_next_2w)
     FROM mart_center_week)
        AS mart_inbound_2w_sum,

    (SELECT SUM(inbound_qty_next_4w)
     FROM expected_po_weekly)
        AS expected_inbound_4w_sum,

    (SELECT SUM(min_inbound_qty_next_4w)
     FROM mart_center_week)
        AS mart_inbound_4w_sum,

    (SELECT SUM(overdue_po_qty)
     FROM expected_po_weekly)
        AS expected_overdue_po_sum,

    (SELECT SUM(min_overdue_po_qty)
     FROM mart_center_week)
        AS mart_overdue_po_sum,

    (SELECT SUM(open_po_qty)
     FROM expected_po_weekly)
    -
    (SELECT SUM(min_open_po_qty)
     FROM mart_center_week)
        AS open_po_qty_difference,

    (SELECT SUM(inbound_qty_next_1w)
     FROM expected_po_weekly)
    -
    (SELECT SUM(min_inbound_qty_next_1w)
     FROM mart_center_week)
        AS inbound_1w_difference,

    (SELECT SUM(inbound_qty_next_2w)
     FROM expected_po_weekly)
    -
    (SELECT SUM(min_inbound_qty_next_2w)
     FROM mart_center_week)
        AS inbound_2w_difference,

    (SELECT SUM(inbound_qty_next_4w)
     FROM expected_po_weekly)
    -
    (SELECT SUM(min_inbound_qty_next_4w)
     FROM mart_center_week)
        AS inbound_4w_difference,

    (SELECT SUM(overdue_po_qty)
     FROM expected_po_weekly)
    -
    (SELECT SUM(min_overdue_po_qty)
     FROM mart_center_week)
        AS overdue_po_difference;
