WITH inventory_mapped AS (
    SELECT
        m.sku_id,
        i.center_id,
        date(
            i.snapshot_datetime,
            '-' || (
                (CAST(strftime('%w', i.snapshot_datetime) AS INTEGER) + 6) % 7
            ) || ' days'
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

raw_stockout_weekly AS (
    SELECT
        sku_id,
        center_id,
        week_start_date,
        MAX(stockout_flag) AS stockout_flag
    FROM inventory_mapped
    GROUP BY
        sku_id,
        center_id,
        week_start_date
),

raw_inventory_weekly AS (
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

    INNER JOIN raw_stockout_weekly AS s
        ON s.sku_id = i.sku_id
       AND s.center_id = i.center_id
       AND s.week_start_date = i.week_start_date

    WHERE i.snapshot_rank = 1
),

mart_center_week AS (
    SELECT
        sku_id,
        center_id,
        week_start_date,

        COUNT(*) AS channel_row_count,

        MIN(on_hand_qty) AS min_on_hand_qty,
        MAX(on_hand_qty) AS max_on_hand_qty,

        MIN(reserved_qty) AS min_reserved_qty,
        MAX(reserved_qty) AS max_reserved_qty,

        MIN(damaged_qty) AS min_damaged_qty,
        MAX(damaged_qty) AS max_damaged_qty,

        MIN(quality_hold_qty) AS min_quality_hold_qty,
        MAX(quality_hold_qty) AS max_quality_hold_qty,

        MIN(available_qty) AS min_available_qty,
        MAX(available_qty) AS max_available_qty,

        MIN(stockout_flag) AS min_stockout_flag,
        MAX(stockout_flag) AS max_stockout_flag

    FROM mart_retail_instock_weekly

    GROUP BY
        sku_id,
        center_id,
        week_start_date
),

row_comparison AS (
    SELECT
        r.sku_id,
        r.center_id,
        r.week_start_date,

        CASE
            WHEN m.sku_id IS NULL THEN 1
            ELSE 0
        END AS missing_in_mart,

        CASE
            WHEN
                r.on_hand_qty <> m.min_on_hand_qty
                OR r.reserved_qty <> m.min_reserved_qty
                OR r.damaged_qty <> m.min_damaged_qty
                OR r.quality_hold_qty <> m.min_quality_hold_qty
                OR r.available_qty <> m.min_available_qty
                OR r.stockout_flag <> m.min_stockout_flag
            THEN 1
            ELSE 0
        END AS value_mismatch

    FROM raw_inventory_weekly AS r

    LEFT JOIN mart_center_week AS m
        ON m.sku_id = r.sku_id
       AND m.center_id = r.center_id
       AND m.week_start_date = r.week_start_date
)

SELECT
    (SELECT COUNT(*) FROM wms_inventory_snapshot)
        AS raw_snapshot_rows,

    (SELECT COUNT(*) FROM inventory_mapped)
        AS mapped_snapshot_rows,

    (SELECT COUNT(*) FROM raw_inventory_weekly)
        AS raw_weekly_keys,

    (SELECT COUNT(*) FROM mart_center_week)
        AS mart_center_week_keys,

    (SELECT COUNT(*)
     FROM mart_center_week
     WHERE channel_row_count <> 3)
        AS unexpected_channel_row_count,

    (SELECT COUNT(*)
     FROM mart_center_week
     WHERE
        min_on_hand_qty <> max_on_hand_qty
        OR min_reserved_qty <> max_reserved_qty
        OR min_damaged_qty <> max_damaged_qty
        OR min_quality_hold_qty <> max_quality_hold_qty
        OR min_available_qty <> max_available_qty
        OR min_stockout_flag <> max_stockout_flag)
        AS channel_repeated_value_mismatch_count,

    (SELECT COUNT(*)
     FROM row_comparison
     WHERE missing_in_mart = 1)
        AS raw_keys_missing_in_mart,

    (SELECT COUNT(*)
     FROM row_comparison
     WHERE value_mismatch = 1)
        AS raw_to_mart_value_mismatch_count,

    (SELECT SUM(on_hand_qty)
     FROM raw_inventory_weekly)
        AS raw_on_hand_qty_sum,

    (SELECT SUM(min_on_hand_qty)
     FROM mart_center_week)
        AS mart_on_hand_qty_sum,

    (SELECT SUM(available_qty)
     FROM raw_inventory_weekly)
        AS raw_available_qty_sum,

    (SELECT SUM(min_available_qty)
     FROM mart_center_week)
        AS mart_available_qty_sum,

    (SELECT SUM(stockout_flag)
     FROM raw_inventory_weekly)
        AS raw_stockout_sum,

    (SELECT SUM(min_stockout_flag)
     FROM mart_center_week)
        AS mart_stockout_sum,

    (SELECT SUM(on_hand_qty)
     FROM raw_inventory_weekly)
    -
    (SELECT SUM(min_on_hand_qty)
     FROM mart_center_week)
        AS on_hand_qty_difference,

    (SELECT SUM(available_qty)
     FROM raw_inventory_weekly)
    -
    (SELECT SUM(min_available_qty)
     FROM mart_center_week)
        AS available_qty_difference,

    (SELECT SUM(stockout_flag)
     FROM raw_inventory_weekly)
    -
    (SELECT SUM(min_stockout_flag)
     FROM mart_center_week)
        AS stockout_difference;
