WITH raw_oms AS (
    SELECT
        COUNT(*) AS raw_order_lines,
        SUM(ordered_qty) AS raw_ordered_qty,
        SUM(fulfilled_qty) AS raw_fulfilled_qty,
        SUM(cancelled_qty) AS raw_cancelled_qty
    FROM oms_sales_orders
),
mapped_oms AS (
    SELECT
        COUNT(*) AS mapped_order_lines,
        SUM(o.ordered_qty) AS mapped_ordered_qty,
        SUM(o.fulfilled_qty) AS mapped_fulfilled_qty,
        SUM(o.cancelled_qty) AS mapped_cancelled_qty
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
mart_sales AS (
    SELECT
        SUM(order_line_count) AS mart_order_lines,
        SUM(ordered_qty_1w) AS mart_ordered_qty,
        SUM(fulfilled_qty_1w) AS mart_fulfilled_qty,
        SUM(cancelled_qty_1w) AS mart_cancelled_qty
    FROM mart_retail_instock_weekly
)
SELECT
    r.raw_order_lines,
    m.mapped_order_lines,
    t.mart_order_lines,

    r.raw_ordered_qty,
    m.mapped_ordered_qty,
    t.mart_ordered_qty,

    r.raw_fulfilled_qty,
    m.mapped_fulfilled_qty,
    t.mart_fulfilled_qty,

    r.raw_cancelled_qty,
    m.mapped_cancelled_qty,
    t.mart_cancelled_qty,

    r.raw_order_lines - m.mapped_order_lines
        AS raw_to_mapping_line_difference,

    m.mapped_order_lines - t.mart_order_lines
        AS mapping_to_mart_line_difference,

    r.raw_ordered_qty - t.mart_ordered_qty
        AS ordered_qty_difference,

    r.raw_fulfilled_qty - t.mart_fulfilled_qty
        AS fulfilled_qty_difference,

    r.raw_cancelled_qty - t.mart_cancelled_qty
        AS cancelled_qty_difference
FROM raw_oms AS r
CROSS JOIN mapped_oms AS m
CROSS JOIN mart_sales AS t;
