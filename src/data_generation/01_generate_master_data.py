"""Generate synthetic product, vendor, and cross-system SKU master data.

This script creates only the reference master data needed by later OMS, WMS,
ERP, and MD synthetic-data generators. It intentionally does not generate any
sales, inventory, purchase-order, receipt, or promotion transactions.
"""

from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 42
REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = REPO_ROOT / "data" / "raw"

VENDOR_COLUMNS = [
    "vendor_id",
    "vendor_name",
    "vendor_country",
    "import_flag",
    "standard_lead_time_days",
    "order_cycle_days",
    "payment_terms",
    "supply_region",
    "reliability_tier",
    "lead_time_profile",
    "fill_rate_profile",
    "vendor_active_flag",
]

SKU_COLUMNS = [
    "sku_id",
    "sku_name",
    "category_l1",
    "category_l2",
    "brand",
    "default_vendor_id",
    "launch_date",
    "discontinue_date",
    "unit_cost",
    "list_price",
    "gross_margin_rate",
    "shelf_life_days",
    "base_weekly_demand",
    "demand_volatility_tier",
    "moq_qty",
    "order_multiple",
    "min_order_amount",
    "active_flag",
]

MAPPING_COLUMNS = [
    "sku_id",
    "source_system",
    "source_sku_code",
    "effective_from",
    "effective_to",
    "mapping_status",
]


def create_vendor_master() -> pd.DataFrame:
    """Create contract attributes and simulation profiles for ten vendors.

    ``standard_lead_time_days`` is a contractual baseline, not an observed
    performance metric. The tier/profile fields are simulation seeds that later
    PO and receipt generators can use to create different supply behaviors.
    Actual average lead time, lead-time deviation, on-time delivery rate, and
    PO fill rate must be calculated later from PO and goods-receipt history.
    """
    rows = [
        ("V001", "Seoul Beauty Labs", "South Korea", 0, 7, 7, "NET30", "Domestic", "HIGH", "STABLE", "HIGH", 1),
        ("V002", "Han River Cosmetics", "South Korea", 0, 10, 7, "NET30", "Domestic", "MEDIUM", "MODERATE", "MEDIUM", 1),
        ("V003", "Korea Wellness Co", "South Korea", 0, 8, 7, "NET30", "Domestic", "HIGH", "STABLE", "HIGH", 1),
        ("V004", "Busan Home Supply", "South Korea", 0, 12, 14, "NET45", "Domestic", "MEDIUM", "MODERATE", "MEDIUM", 1),
        ("V005", "Incheon Living Goods", "South Korea", 0, 6, 7, "NET30", "Domestic", "HIGH", "STABLE", "HIGH", 1),
        ("V006", "Sakura Trading", "Japan", 1, 28, 14, "NET45", "East Asia", "MEDIUM", "VARIABLE", "MEDIUM", 1),
        ("V007", "Pacific Health Imports", "United States", 1, 42, 28, "NET60", "North America", "LOW", "HIGH_VARIANCE", "LOW", 1),
        ("V008", "Nordic Lifestyle AB", "Sweden", 1, 35, 28, "NET60", "Europe", "MEDIUM", "VARIABLE", "MEDIUM", 1),
        ("V009", "Shenzhen Smart Living", "China", 1, 24, 14, "NET45", "East Asia", "MEDIUM", "VARIABLE", "HIGH", 1),
        ("V010", "Green Home Partners", "South Korea", 0, 9, 7, "NET30", "Domestic", "LOW", "HIGH_VARIANCE", "LOW", 1),
    ]
    return pd.DataFrame(rows, columns=VENDOR_COLUMNS)


def create_sku_master(rng: np.random.Generator) -> pd.DataFrame:
    """Create forty synthetic SKUs spanning four top-level categories."""
    product_catalog = {
        "Beauty": [
            ("Skincare", "Hydrating Toner"),
            ("Skincare", "Barrier Cream"),
            ("Skincare", "Vitamin Serum"),
            ("Skincare", "Daily Sunscreen"),
            ("Haircare", "Repair Shampoo"),
            ("Haircare", "Volume Conditioner"),
            ("Haircare", "Scalp Treatment"),
            ("Makeup", "Velvet Lip Tint"),
            ("Makeup", "Glow Cushion"),
            ("Makeup", "Brow Pencil"),
        ],
        "Health": [
            ("Supplements", "Daily Multivitamin"),
            ("Supplements", "Vitamin C Tablets"),
            ("Supplements", "Omega Three Capsules"),
            ("Supplements", "Probiotic Sachets"),
            ("Personal Care", "Cooling Body Patch"),
            ("Personal Care", "Hand Sanitizer"),
            ("Personal Care", "Dental Floss Set"),
            ("Fitness", "Resistance Band"),
            ("Fitness", "Foam Roller"),
            ("Fitness", "Protein Shaker"),
        ],
        "Home": [
            ("Cleaning", "Multi Surface Cleaner"),
            ("Cleaning", "Laundry Detergent"),
            ("Cleaning", "Kitchen Cleaning Wipes"),
            ("Cleaning", "Dishwasher Tablets"),
            ("Kitchen", "Glass Storage Set"),
            ("Kitchen", "Silicone Utensil Set"),
            ("Kitchen", "Reusable Food Bags"),
            ("Bedding", "Cotton Pillowcase Set"),
            ("Bedding", "Cooling Blanket"),
            ("Bedding", "Mattress Protector"),
        ],
        "Lifestyle": [
            ("Travel", "Packing Cube Set"),
            ("Travel", "Travel Bottle Kit"),
            ("Travel", "Foldable Neck Pillow"),
            ("Stationery", "Weekly Planner"),
            ("Stationery", "Gel Pen Set"),
            ("Stationery", "Desk Organizer"),
            ("Outdoor", "Insulated Tumbler"),
            ("Outdoor", "Compact Picnic Mat"),
            ("Outdoor", "Portable Lantern"),
            ("Outdoor", "Lightweight Daypack"),
        ],
    }

    category_settings = {
        "Beauty": {
            "brands": ["Aurelia", "Mellow Dew", "Pure Seoul"],
            "vendors": ["V001", "V002", "V006"],
            "shelf_life_days": [730, 900, 1095],
            "demand_range": (45, 180),
        },
        "Health": {
            "brands": ["Wellnest", "VitaRoot", "Core Balance"],
            "vendors": ["V003", "V007"],
            "shelf_life_days": [365, 540, 730],
            "demand_range": (30, 150),
        },
        "Home": {
            "brands": ["Neat Day", "Home Harbor", "Green Habit"],
            "vendors": ["V004", "V005", "V010"],
            "shelf_life_days": [730, 1095, 1825],
            "demand_range": (25, 130),
        },
        "Lifestyle": {
            "brands": ["Roamly", "Paper Grove", "Urban Trail"],
            "vendors": ["V005", "V008", "V009"],
            "shelf_life_days": [1095, 1825, 3650],
            "demand_range": (20, 110),
        },
    }

    order_multiples = np.array([6, 12, 24])
    launch_start = np.datetime64("2021-01-01")
    launch_span_days = 4 * 365
    rows: list[dict[str, object]] = []

    sku_number = 1
    for category_l1, products in product_catalog.items():
        settings = category_settings[category_l1]
        for category_l2, product_name in products:
            sku_id = f"SKU{sku_number:04d}"
            order_multiple = int(rng.choice(order_multiples))
            moq_qty = order_multiple * int(rng.integers(2, 7))
            list_price = float(rng.choice(np.arange(9_900, 59_901, 1_000)))
            target_margin = float(rng.uniform(0.32, 0.62))
            unit_cost = round(list_price * (1 - target_margin), 2)
            gross_margin_rate = round((list_price - unit_cost) / list_price, 4)
            launch_date = launch_start + np.timedelta64(
                int(rng.integers(0, launch_span_days)), "D"
            )

            # These two columns are simulation seeds, not normal operational
            # Product Master fields. Later generators use them to create demand
            # level and variability differences between synthetic SKUs.
            base_weekly_demand = int(rng.integers(*settings["demand_range"]))
            demand_volatility_tier = str(
                rng.choice(["LOW", "MEDIUM", "HIGH"], p=[0.30, 0.50, 0.20])
            )

            rows.append(
                {
                    "sku_id": sku_id,
                    "sku_name": f"{settings['brands'][sku_number % 3]} {product_name}",
                    "category_l1": category_l1,
                    "category_l2": category_l2,
                    "brand": settings["brands"][sku_number % 3],
                    "default_vendor_id": str(rng.choice(settings["vendors"])),
                    "launch_date": pd.Timestamp(launch_date).strftime("%Y-%m-%d"),
                    "discontinue_date": "",
                    "unit_cost": unit_cost,
                    "list_price": list_price,
                    "gross_margin_rate": gross_margin_rate,
                    "shelf_life_days": int(rng.choice(settings["shelf_life_days"])),
                    "base_weekly_demand": base_weekly_demand,
                    "demand_volatility_tier": demand_volatility_tier,
                    "moq_qty": moq_qty,
                    "order_multiple": order_multiple,
                    "min_order_amount": round(unit_cost * moq_qty, 2),
                    "active_flag": 1,
                }
            )
            sku_number += 1

    return pd.DataFrame(rows, columns=SKU_COLUMNS)


def create_sku_code_mapping(sku_master: pd.DataFrame) -> pd.DataFrame:
    """Create one long-format code mapping row per SKU and source system."""
    code_templates = {
        "OMS": "OMS-P{number:05d}",
        "WMS": "WMS-L{number:05d}",
        "ERP": "ERP-I{number:05d}",
        "MD": "MD-M{number:05d}",
    }
    rows: list[dict[str, object]] = []

    for number, sku in enumerate(sku_master.itertuples(index=False), start=1):
        for source_system, template in code_templates.items():
            rows.append(
                {
                    "sku_id": sku.sku_id,
                    "source_system": source_system,
                    "source_sku_code": template.format(number=number),
                    "effective_from": sku.launch_date,
                    "effective_to": sku.discontinue_date,
                    "mapping_status": "VALID",
                }
            )

    return pd.DataFrame(rows, columns=MAPPING_COLUMNS)


def validate_master_data(
    sku_master: pd.DataFrame,
    vendor_master: pd.DataFrame,
    sku_code_mapping: pd.DataFrame,
) -> None:
    """Fail fast if a generated master violates the MVP contract."""
    assert list(sku_master.columns) == SKU_COLUMNS
    assert list(vendor_master.columns) == VENDOR_COLUMNS
    assert list(sku_code_mapping.columns) == MAPPING_COLUMNS
    assert len(sku_master) == 40
    assert len(vendor_master) == 10
    assert len(sku_code_mapping) == 160
    assert sku_master["sku_id"].is_unique
    assert vendor_master["vendor_id"].is_unique
    assert sku_master["category_l1"].nunique() == 4
    assert set(sku_code_mapping["source_system"]) == {"OMS", "WMS", "ERP", "MD"}
    assert not sku_code_mapping.duplicated(
        ["source_system", "source_sku_code", "effective_from"]
    ).any()
    assert sku_code_mapping.groupby("sku_id")["source_system"].nunique().eq(4).all()
    assert set(sku_master["default_vendor_id"]).issubset(
        set(vendor_master["vendor_id"])
    )

    prohibited_vendor_metrics = {
        "vendor_avg_lead_time",
        "vendor_lead_time_std",
        "on_time_delivery_rate",
        "po_fill_rate",
    }
    assert prohibited_vendor_metrics.isdisjoint(vendor_master.columns)


def save_csv(dataframe: pd.DataFrame, filename: str) -> Path:
    """Save a DataFrame as a UTF-8 CSV and return its path."""
    output_path = RAW_DATA_DIR / filename
    dataframe.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def main() -> None:
    """Generate, validate, and save the three synthetic master datasets."""
    rng = np.random.default_rng(RANDOM_SEED)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    vendor_master = create_vendor_master()
    sku_master = create_sku_master(rng)
    sku_code_mapping = create_sku_code_mapping(sku_master)

    validate_master_data(sku_master, vendor_master, sku_code_mapping)

    outputs = [
        (sku_master, save_csv(sku_master, "sku_master.csv")),
        (vendor_master, save_csv(vendor_master, "vendor_master.csv")),
        (sku_code_mapping, save_csv(sku_code_mapping, "sku_code_mapping.csv")),
    ]

    for dataframe, output_path in outputs:
        print(f"Created {output_path} ({len(dataframe)} rows)")


if __name__ == "__main__":
    main()
