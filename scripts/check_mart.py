from pathlib import Path
import sqlite3

db_path = Path("data/database/retail_instock.db")

with sqlite3.connect(db_path) as conn:
    cur = conn.cursor()

    table = "mart_retail_instock_weekly"

    row_count = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    unique_key_count = cur.execute(f"""
        SELECT COUNT(*)
        FROM (
            SELECT sku_id, channel_id, center_id, week_start_date
            FROM {table}
            GROUP BY sku_id, channel_id, center_id, week_start_date
        )
    """).fetchone()[0]

    stockout_count = cur.execute(f"SELECT SUM(stockout_flag) FROM {table}").fetchone()[0]
    promo_count = cur.execute(f"SELECT SUM(promo_flag) FROM {table}").fetchone()[0]
    partial_count = cur.execute(f"SELECT SUM(partial_fulfillment_flag) FROM {table}").fetchone()[0]
    inbound_4w_sum = cur.execute(f"SELECT SUM(inbound_qty_next_4w) FROM {table}").fetchone()[0]
    sales_sum = cur.execute(f"SELECT SUM(fulfilled_qty_1w) FROM {table}").fetchone()[0]

    print(f"row_count: {row_count}")
    print(f"unique_key_count: {unique_key_count}")
    print(f"duplicate_key_count: {row_count - unique_key_count}")
    print(f"stockout_count: {stockout_count}")
    print(f"promo_count: {promo_count}")
    print(f"partial_fulfillment_count: {partial_count}")
    print(f"inbound_qty_next_4w_sum: {inbound_4w_sum}")
    print(f"fulfilled_qty_1w_sum: {sales_sum}")
