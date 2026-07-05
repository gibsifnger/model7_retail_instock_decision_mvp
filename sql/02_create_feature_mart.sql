-- Weekly integrated source mart for the Retail InStock Decision MVP.
-- Grain: sku_id x channel_id x center_id x week_start_date.
-- This script intentionally does not create targets, lag/rolling features,
-- model outputs, or reorder decisions.

DROP TABLE IF EXISTS mart_retail_instock_weekly;

CREATE TABLE mart_retail_instock_weekly AS
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
sku_center_week AS (
    SELECT
        s.sku_id,
        f.center_id,
        w.week_start_date
    FROM sku_master AS s
    CROSS JOIN centers AS f
    CROSS JOIN weeks AS w
),

-- Map OMS codes to the standard SKU and aggregate order lines weekly.
sales_mapped AS (
    SELECT
        m.sku_id,
        o.channel_id,
        o.fulfillment_center_id AS center_id,
        date(
            o.order_datetime,
            '-' || ((CAST(strftime('%w', o.order_datetime) AS INTEGER) + 6) % 7)
                || ' days'
        ) AS week_start_date,
        o.ordered_qty,
        o.fulfilled_qty,
        o.cancelled_qty,
        o.unit_selling_price
    FROM oms_sales_orders AS o
    INNER JOIN sku_code_mapping AS m
        ON m.source_system = 'OMS'
       AND m.source_sku_code = o.oms_sku_code
       AND m.mapping_status = 'VALID'
       AND date(o.order_datetime) >= date(m.effective_from)
       AND (
            NULLIF(m.effective_to, '') IS NULL
            OR date(o.order_datetime) <= date(m.effective_to)
       )
),
sales_weekly AS (
    SELECT
        sku_id,
        channel_id,
        center_id,
        week_start_date,
        SUM(ordered_qty) AS ordered_qty_1w,
        SUM(fulfilled_qty) AS fulfilled_qty_1w,
        SUM(cancelled_qty) AS cancelled_qty_1w,
        AVG(unit_selling_price) AS avg_selling_price,
        COUNT(*) AS order_line_count,
        MAX(CASE WHEN fulfilled_qty < ordered_qty THEN 1 ELSE 0 END)
            AS partial_fulfillment_flag
    FROM sales_mapped
    GROUP BY sku_id, channel_id, center_id, week_start_date
),

-- Use the latest WMS snapshot in each week for quantities. If multiple
-- snapshots exist, stockout_flag still records whether any snapshot stocked out.
inventory_mapped AS (
    SELECT
        m.sku_id,
        i.center_id,
        date(
            i.snapshot_datetime,
            '-' || ((CAST(strftime('%w', i.snapshot_datetime) AS INTEGER) + 6) % 7)
                || ' days'
        ) AS week_start_date,
        i.snapshot_datetime,
        i.on_hand_qty,
        i.reserved_qty,
        i.damaged_qty,
        i.quality_hold_qty,
        i.available_qty,
        i.stockout_flag,
        ROW_NUMBER() OVER (
            PARTITION BY
                m.sku_id,
                i.center_id,
                date(
                    i.snapshot_datetime,
                    '-' || (
                        (CAST(strftime('%w', i.snapshot_datetime) AS INTEGER) + 6) % 7
                    ) || ' days'
                )
            ORDER BY datetime(i.snapshot_datetime) DESC
        ) AS snapshot_rank
    FROM wms_inventory_snapshot AS i
    INNER JOIN sku_code_mapping AS m
        ON m.source_system = 'WMS'
       AND m.source_sku_code = i.wms_sku_code
       AND m.mapping_status = 'VALID'
       AND date(i.snapshot_datetime) >= date(m.effective_from)
       AND (
            NULLIF(m.effective_to, '') IS NULL
            OR date(i.snapshot_datetime) <= date(m.effective_to)
       )
),
inventory_stockout_weekly AS (
    SELECT
        sku_id,
        center_id,
        week_start_date,
        MAX(stockout_flag) AS stockout_flag
    FROM inventory_mapped
    GROUP BY sku_id, center_id, week_start_date
),
inventory_weekly AS (
    SELECT
        i.sku_id,
        i.center_id,
        i.week_start_date,
        i.on_hand_qty,
        i.reserved_qty,
        i.damaged_qty,
        i.quality_hold_qty,
        i.available_qty,
        s.stockout_flag
    FROM inventory_mapped AS i
    INNER JOIN inventory_stockout_weekly AS s
        ON s.sku_id = i.sku_id
       AND s.center_id = i.center_id
       AND s.week_start_date = i.week_start_date
    WHERE i.snapshot_rank = 1
),

-- Map ERP and WMS codes independently, then connect receipts to their PO line.
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
        r.accepted_qty
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
po_position_by_week AS (
    SELECT
        g.sku_id,
        g.center_id,
        g.week_start_date,
        p.po_id,
        p.po_line_id,
        p.promised_delivery_date,
        p.confirmed_qty,
        COALESCE(SUM(
            CASE
                WHEN r.receipt_datetime < datetime(g.week_start_date, '+8 hours')
                THEN r.accepted_qty
                ELSE 0
            END
        ), 0) AS accepted_qty_as_of_week
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
po_weekly AS (
    SELECT
        sku_id,
        center_id,
        week_start_date,
        SUM(CASE
            WHEN promised_delivery_date >= week_start_date
             AND promised_delivery_date < date(week_start_date, '+7 days')
            THEN remaining_qty ELSE 0
        END) AS inbound_qty_next_1w,
        SUM(CASE
            WHEN promised_delivery_date >= week_start_date
             AND promised_delivery_date < date(week_start_date, '+14 days')
            THEN remaining_qty ELSE 0
        END) AS inbound_qty_next_2w,
        SUM(CASE
            WHEN promised_delivery_date >= week_start_date
             AND promised_delivery_date < date(week_start_date, '+28 days')
            THEN remaining_qty ELSE 0
        END) AS inbound_qty_next_4w,
        SUM(remaining_qty) AS open_po_qty,
        SUM(CASE
            WHEN promised_delivery_date < week_start_date
            THEN remaining_qty ELSE 0
        END) AS overdue_po_qty
    FROM po_remaining_by_week
    GROUP BY sku_id, center_id, week_start_date
),

-- Select one promotion per SKU/channel/week: business priority first, then
-- the deepest planned discount as the tie-breaker.
promotion_candidates AS (
    SELECT
        g.sku_id,
        g.channel_id,
        g.week_start_date,
        p.promo_type,
        p.planned_discount_rate AS promo_depth,
        p.planned_promo_price,
        p.promo_priority,
        ROW_NUMBER() OVER (
            PARTITION BY g.sku_id, g.channel_id, g.week_start_date
            ORDER BY
                CASE p.promo_priority
                    WHEN 'HIGH' THEN 1
                    WHEN 'MEDIUM' THEN 2
                    WHEN 'LOW' THEN 3
                    ELSE 4
                END,
                p.planned_discount_rate DESC,
                p.promotion_id
        ) AS promo_rank
    FROM (
        SELECT DISTINCT sku_id, channel_id, week_start_date
        FROM base_grid
    ) AS g
    INNER JOIN sku_code_mapping AS m
        ON m.sku_id = g.sku_id
       AND m.source_system = 'MD'
       AND m.mapping_status = 'VALID'
    INNER JOIN md_promotion_calendar AS p
        ON p.md_sku_code = m.source_sku_code
       AND p.channel_id = g.channel_id
       AND date(p.promo_start_date) <= date(g.week_start_date, '+6 days')
       AND date(p.promo_end_date) >= g.week_start_date
       AND date(p.promo_start_date) >= date(m.effective_from)
       AND (
            NULLIF(m.effective_to, '') IS NULL
            OR date(p.promo_end_date) <= date(m.effective_to)
       )
),
promotion_weekly AS (
    SELECT
        sku_id,
        channel_id,
        week_start_date,
        1 AS promo_flag,
        promo_type,
        promo_depth,
        planned_promo_price,
        promo_priority
    FROM promotion_candidates
    WHERE promo_rank = 1
),

-- Only approved, in-scope manual rules are active in the integrated mart.
override_candidates AS (
    SELECT
        g.sku_id,
        g.channel_id,
        g.center_id,
        g.week_start_date,
        o.override_type,
        o.override_reason,
        ROW_NUMBER() OVER (
            PARTITION BY
                g.sku_id,
                g.channel_id,
                g.center_id,
                g.week_start_date
            ORDER BY datetime(o.created_at) DESC, o.override_id
        ) AS override_rank
    FROM base_grid AS g
    INNER JOIN manual_overrides AS o
        ON o.sku_id = g.sku_id
       AND (NULLIF(o.channel_id, '') IS NULL OR o.channel_id = g.channel_id)
       AND (NULLIF(o.center_id, '') IS NULL OR o.center_id = g.center_id)
       AND g.week_start_date BETWEEN date(o.effective_from) AND date(o.effective_to)
       AND o.approval_status = 'APPROVED'
),
manual_override_weekly AS (
    SELECT
        sku_id,
        channel_id,
        center_id,
        week_start_date,
        1 AS manual_override_flag,
        override_type,
        override_reason
    FROM override_candidates
    WHERE override_rank = 1
)
SELECT
    g.sku_id,
    g.channel_id,
    g.center_id,
    g.week_start_date,

    s.category_l1,
    s.category_l2,
    s.brand,
    s.default_vendor_id,
    s.launch_date,
    s.discontinue_date,
    s.unit_cost,
    s.list_price,
    s.gross_margin_rate,
    s.shelf_life_days,
    s.moq_qty,
    s.order_multiple,
    s.min_order_amount,
    s.active_flag,

    v.vendor_country,
    v.import_flag,
    v.standard_lead_time_days,
    v.order_cycle_days,
    v.payment_terms,
    v.supply_region,
    v.reliability_tier,
    v.lead_time_profile,
    v.fill_rate_profile,
    v.vendor_active_flag,

    COALESCE(sw.ordered_qty_1w, 0) AS ordered_qty_1w,
    COALESCE(sw.fulfilled_qty_1w, 0) AS fulfilled_qty_1w,
    COALESCE(sw.cancelled_qty_1w, 0) AS cancelled_qty_1w,
    sw.avg_selling_price,
    COALESCE(sw.order_line_count, 0) AS order_line_count,
    COALESCE(sw.partial_fulfillment_flag, 0) AS partial_fulfillment_flag,

    COALESCE(iw.on_hand_qty, 0) AS on_hand_qty,
    COALESCE(iw.reserved_qty, 0) AS reserved_qty,
    COALESCE(iw.damaged_qty, 0) AS damaged_qty,
    COALESCE(iw.quality_hold_qty, 0) AS quality_hold_qty,
    COALESCE(iw.available_qty, 0) AS available_qty,
    COALESCE(iw.stockout_flag, 0) AS stockout_flag,

    COALESCE(pw.inbound_qty_next_1w, 0) AS inbound_qty_next_1w,
    COALESCE(pw.inbound_qty_next_2w, 0) AS inbound_qty_next_2w,
    COALESCE(pw.inbound_qty_next_4w, 0) AS inbound_qty_next_4w,
    COALESCE(pw.open_po_qty, 0) AS open_po_qty,
    COALESCE(pw.overdue_po_qty, 0) AS overdue_po_qty,

    COALESCE(pr.promo_flag, 0) AS promo_flag,
    pr.promo_type,
    pr.promo_depth,
    pr.planned_promo_price,
    pr.promo_priority,

    COALESCE(mo.manual_override_flag, 0) AS manual_override_flag,
    mo.override_type,
    mo.override_reason
FROM base_grid AS g
INNER JOIN sku_master AS s
    ON s.sku_id = g.sku_id
LEFT JOIN vendor_master AS v
    ON v.vendor_id = s.default_vendor_id
LEFT JOIN sales_weekly AS sw
    ON sw.sku_id = g.sku_id
   AND sw.channel_id = g.channel_id
   AND sw.center_id = g.center_id
   AND sw.week_start_date = g.week_start_date
LEFT JOIN inventory_weekly AS iw
    ON iw.sku_id = g.sku_id
   AND iw.center_id = g.center_id
   AND iw.week_start_date = g.week_start_date
LEFT JOIN po_weekly AS pw
    ON pw.sku_id = g.sku_id
   AND pw.center_id = g.center_id
   AND pw.week_start_date = g.week_start_date
LEFT JOIN promotion_weekly AS pr
    ON pr.sku_id = g.sku_id
   AND pr.channel_id = g.channel_id
   AND pr.week_start_date = g.week_start_date
LEFT JOIN manual_override_weekly AS mo
    ON mo.sku_id = g.sku_id
   AND mo.channel_id = g.channel_id
   AND mo.center_id = g.center_id
   AND mo.week_start_date = g.week_start_date;

CREATE UNIQUE INDEX idx_mart_retail_instock_weekly_grain
ON mart_retail_instock_weekly (
    sku_id,
    channel_id,
    center_id,
    week_start_date
);

-- Validation summary for interactive execution.
SELECT
    COUNT(*) AS mart_row_count,
    COUNT(*) - (
        SELECT COUNT(*)
        FROM (
            SELECT sku_id, channel_id, center_id, week_start_date
            FROM mart_retail_instock_weekly
            GROUP BY sku_id, channel_id, center_id, week_start_date
        ) AS unique_grain
    ) AS primary_key_duplicate_count,
    SUM(stockout_flag) AS stockout_flag_sum,
    SUM(promo_flag) AS promo_flag_sum,
    SUM(inbound_qty_next_4w) AS inbound_qty_next_4w_sum
FROM mart_retail_instock_weekly;
