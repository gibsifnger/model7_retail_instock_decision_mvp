WITH sample_key AS (
    SELECT
        sku_id,
        center_id,
        week_start_date
    FROM mart_retail_instock_weekly
    WHERE available_qty > 0
    GROUP BY
        sku_id,
        center_id,
        week_start_date
    HAVING
        COUNT(*) = 3
        AND SUM(ordered_qty_1w) > 0
    ORDER BY week_start_date DESC
    LIMIT 1
)
SELECT
    m.sku_id,
    m.channel_id,
    m.center_id,
    m.week_start_date,
    m.ordered_qty_1w,
    m.fulfilled_qty_1w,
    m.available_qty,
    m.open_po_qty,
    m.stockout_flag
FROM mart_retail_instock_weekly AS m
INNER JOIN sample_key AS k
    ON k.sku_id = m.sku_id
   AND k.center_id = m.center_id
   AND k.week_start_date = m.week_start_date
ORDER BY m.channel_id;
