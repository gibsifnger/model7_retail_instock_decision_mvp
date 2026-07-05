"""Generate synthetic OMS, WMS, ERP, MD, and manual source data for 2025.

The outputs intentionally retain different source-system grains. Feature tables
and ML-ready data at the SKU x channel x center x week decision grain are not
created in this step.
"""

from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 20250705
REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = REPO_ROOT / "data" / "raw"

WEEK_STARTS = pd.date_range("2025-01-06", "2025-12-29", freq="W-MON")
DATA_END = pd.Timestamp("2025-12-31 23:59:59")
CHANNELS = ["ONLINE_MALL", "ROCKET_DELIVERY", "GLOBAL_MALL"]
CENTERS = ["MFC_GANGNAM", "FC_DONGTAN", "FC_INCHEON"]
SOURCE_SYSTEMS = {"OMS", "WMS", "ERP", "MD"}

CHANNEL_SHARES = {
    "ONLINE_MALL": 0.45,
    "ROCKET_DELIVERY": 0.40,
    "GLOBAL_MALL": 0.15,
}
CENTER_SHARES = {
    "MFC_GANGNAM": 0.25,
    "FC_DONGTAN": 0.45,
    "FC_INCHEON": 0.30,
}
VOLATILITY_SIGMA = {"LOW": 0.10, "MEDIUM": 0.22, "HIGH": 0.38}
PROMO_UPLIFT = {
    "PRICE_DISCOUNT": 0.45,
    "COUPON": 0.30,
    "SPECIAL_EXHIBITION": 0.55,
}

OMS_COLUMNS = [
    "order_id",
    "order_line_id",
    "order_datetime",
    "channel_id",
    "fulfillment_center_id",
    "oms_sku_code",
    "ordered_qty",
    "fulfilled_qty",
    "cancelled_qty",
    "unit_selling_price",
    "order_status",
]
INVENTORY_COLUMNS = [
    "snapshot_datetime",
    "center_id",
    "wms_sku_code",
    "on_hand_qty",
    "reserved_qty",
    "damaged_qty",
    "quality_hold_qty",
    "available_qty",
    "stockout_flag",
]
PO_COLUMNS = [
    "po_id",
    "po_line_id",
    "po_created_date",
    "vendor_id",
    "erp_sku_code",
    "center_id",
    "ordered_qty",
    "confirmed_qty",
    "promised_delivery_date",
    "po_status",
    "cancelled_qty",
    "last_updated_at",
]
RECEIPT_COLUMNS = [
    "receipt_id",
    "receipt_line_id",
    "po_id",
    "po_line_id",
    "receipt_datetime",
    "center_id",
    "wms_sku_code",
    "received_qty",
    "accepted_qty",
    "rejected_qty",
    "receipt_status",
]
PROMOTION_COLUMNS = [
    "promotion_id",
    "md_sku_code",
    "channel_id",
    "promo_type",
    "promo_start_date",
    "promo_end_date",
    "planned_discount_rate",
    "planned_promo_price",
    "promo_priority",
    "calendar_updated_at",
]
OVERRIDE_COLUMNS = [
    "override_id",
    "sku_id",
    "channel_id",
    "center_id",
    "effective_from",
    "effective_to",
    "override_type",
    "override_value",
    "override_reason",
    "created_at",
    "created_by",
    "approval_status",
]


def load_master_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and validate the three master inputs created by step 01."""
    input_paths = {
        "sku": RAW_DATA_DIR / "sku_master.csv",
        "vendor": RAW_DATA_DIR / "vendor_master.csv",
        "mapping": RAW_DATA_DIR / "sku_code_mapping.csv",
    }
    missing = [str(path) for path in input_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing master input(s). Run 01_generate_master_data.py first: "
            + ", ".join(missing)
        )

    sku_master = pd.read_csv(input_paths["sku"])
    vendor_master = pd.read_csv(input_paths["vendor"])
    mapping = pd.read_csv(input_paths["mapping"], keep_default_na=False)

    if len(sku_master) != 40 or len(vendor_master) != 10:
        raise ValueError("Expected 40 SKUs and 10 vendors in the master inputs.")
    if set(mapping["source_system"]) != SOURCE_SYSTEMS:
        raise ValueError("SKU mapping must contain OMS, WMS, ERP, and MD rows.")
    if not mapping.groupby("sku_id")["source_system"].nunique().eq(4).all():
        raise ValueError("Every SKU must have exactly one code in each source system.")

    return sku_master, vendor_master, mapping


def build_code_lookup(mapping: pd.DataFrame) -> dict[tuple[str, str], str]:
    """Return {(sku_id, source_system): source_sku_code}."""
    valid_mapping = mapping.loc[mapping["mapping_status"] == "VALID"]
    return {
        (row.sku_id, row.source_system): row.source_sku_code
        for row in valid_mapping.itertuples(index=False)
    }


def round_up_to_multiple(quantity: float, multiple: int) -> int:
    """Round a non-negative quantity up to a valid order multiple."""
    return int(np.ceil(max(quantity, 0) / multiple) * multiple)


def generate_promotions(
    sku_master: pd.DataFrame,
    code_lookup: dict[tuple[str, str], str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Create planned 7- or 14-day promotions for a subset of SKUs."""
    selected_skus = rng.choice(sku_master["sku_id"], size=26, replace=False)
    sku_by_id = sku_master.set_index("sku_id")
    rows: list[dict[str, object]] = []
    promotion_number = 1

    for sku_id in selected_skus:
        sku = sku_by_id.loc[sku_id]
        promotion_count = int(rng.integers(1, 4))
        start_indices = rng.choice(np.arange(2, 49), size=promotion_count, replace=False)

        for start_index in sorted(start_indices):
            promo_type = str(rng.choice(list(PROMO_UPLIFT)))
            duration_days = int(rng.choice([7, 7, 7, 14]))
            discount_range = {
                "PRICE_DISCOUNT": (0.10, 0.30),
                "COUPON": (0.05, 0.20),
                "SPECIAL_EXHIBITION": (0.08, 0.25),
            }[promo_type]
            discount_rate = round(float(rng.uniform(*discount_range)), 2)
            start_date = WEEK_STARTS[int(start_index)]
            end_date = start_date + pd.Timedelta(days=duration_days - 1)
            calendar_updated = start_date - pd.Timedelta(
                days=int(rng.integers(14, 43))
            )
            channel_count = int(rng.choice([1, 1, 2]))
            promo_channels = rng.choice(CHANNELS, size=channel_count, replace=False)
            promotion_id = f"PROMO{promotion_number:04d}"

            for channel_id in promo_channels:
                rows.append(
                    {
                        "promotion_id": promotion_id,
                        "md_sku_code": code_lookup[(sku_id, "MD")],
                        "channel_id": channel_id,
                        "promo_type": promo_type,
                        "promo_start_date": start_date.strftime("%Y-%m-%d"),
                        "promo_end_date": end_date.strftime("%Y-%m-%d"),
                        "planned_discount_rate": discount_rate,
                        "planned_promo_price": round(
                            float(sku["list_price"]) * (1 - discount_rate), 2
                        ),
                        "promo_priority": str(
                            rng.choice(["HIGH", "MEDIUM", "LOW"], p=[0.2, 0.6, 0.2])
                        ),
                        "calendar_updated_at": calendar_updated.strftime(
                            "%Y-%m-%d 09:00:00"
                        ),
                    }
                )
            promotion_number += 1

    return pd.DataFrame(rows, columns=PROMOTION_COLUMNS).sort_values(
        ["promo_start_date", "promotion_id", "channel_id"], ignore_index=True
    )


def build_promotion_lookup(
    promotions: pd.DataFrame,
    mapping: pd.DataFrame,
) -> dict[tuple[str, str, pd.Timestamp], tuple[float, float]]:
    """Map SKU/channel/week to its strongest promotion uplift and discount."""
    md_to_sku = dict(
        mapping.loc[mapping["source_system"] == "MD", ["source_sku_code", "sku_id"]]
        .itertuples(index=False, name=None)
    )
    lookup: dict[tuple[str, str, pd.Timestamp], tuple[float, float]] = {}

    for promo in promotions.itertuples(index=False):
        sku_id = md_to_sku[promo.md_sku_code]
        promo_start = pd.Timestamp(promo.promo_start_date)
        promo_end = pd.Timestamp(promo.promo_end_date)
        for week_start in WEEK_STARTS:
            week_end = week_start + pd.Timedelta(days=6)
            if promo_start <= week_end and promo_end >= week_start:
                base_uplift = PROMO_UPLIFT[promo.promo_type]
                discount_effect = float(promo.planned_discount_rate) * 0.8
                value = (base_uplift + discount_effect, promo.planned_discount_rate)
                key = (sku_id, promo.channel_id, week_start)
                if key not in lookup or value[0] > lookup[key][0]:
                    lookup[key] = value

    return lookup


def profile_parameters(vendor: pd.Series) -> tuple[tuple[int, int], tuple[float, float], float]:
    """Translate vendor simulation profiles into delay, fill, and split behavior."""
    delay_ranges = {
        "STABLE": (-1, 3),
        "MODERATE": (-1, 7),
        "VARIABLE": (-2, 15),
        "HIGH_VARIANCE": (-3, 29),
    }
    fill_ranges = {
        "HIGH": (0.94, 1.01),
        "MEDIUM": (0.78, 0.98),
        "LOW": (0.58, 0.88),
    }
    split_probability = {"HIGH": 0.10, "MEDIUM": 0.25, "LOW": 0.45}
    return (
        delay_ranges[str(vendor["lead_time_profile"])],
        fill_ranges[str(vendor["fill_rate_profile"])],
        split_probability[str(vendor["reliability_tier"])],
    )


def generate_purchase_orders_and_receipts(
    sku_master: pd.DataFrame,
    vendor_master: pd.DataFrame,
    code_lookup: dict[tuple[str, str], str],
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create linked PO lines and receipts using vendor behavior profiles."""
    vendors = vendor_master.set_index("vendor_id")
    po_rows: list[dict[str, object]] = []
    receipt_rows: list[dict[str, object]] = []
    po_number = 1
    receipt_number = 1

    for sku in sku_master.itertuples(index=False):
        vendor = vendors.loc[sku.default_vendor_id]
        delay_range, fill_range, split_probability = profile_parameters(vendor)

        for center_id in CENTERS:
            expected_center_demand = float(sku.base_weekly_demand) * CENTER_SHARES[center_id]
            for week_index in range(0, len(WEEK_STARTS), 4):
                # A skipped review cycle creates realistic supply gaps and stockout risk.
                skip_probability = {"HIGH": 0.04, "MEDIUM": 0.09, "LOW": 0.16}[
                    str(vendor["reliability_tier"])
                ]
                if rng.random() < skip_probability:
                    continue

                po_created = WEEK_STARTS[week_index] + pd.Timedelta(
                    days=int(rng.integers(0, 3))
                )
                coverage_weeks = float(rng.uniform(4.2, 5.8))
                raw_order_qty = max(
                    expected_center_demand * coverage_weeks,
                    float(sku.moq_qty),
                )
                ordered_qty = round_up_to_multiple(raw_order_qty, int(sku.order_multiple))
                ordered_qty = max(ordered_qty, int(sku.moq_qty))
                cancellation_probability = {
                    "HIGH": 0.005,
                    "MEDIUM": 0.015,
                    "LOW": 0.035,
                }[str(vendor["reliability_tier"])]
                if rng.random() < cancellation_probability:
                    confirmed_qty = 0
                else:
                    confirmed_ratio = float(rng.uniform(*fill_range))
                    confirmed_qty = min(
                        ordered_qty,
                        int(np.floor(ordered_qty * confirmed_ratio)),
                    )
                    confirmed_qty = max(0, confirmed_qty)
                cancelled_qty = ordered_qty - confirmed_qty
                promised_date = po_created + pd.Timedelta(
                    days=int(vendor["standard_lead_time_days"])
                )
                po_id = f"PO{po_number:06d}"
                po_line_id = "001"

                completion_probability = {
                    "HIGH": 0.82,
                    "MEDIUM": 0.62,
                    "LOW": 0.38,
                }[str(vendor["fill_rate_profile"])]
                if rng.random() < completion_probability:
                    actual_fill_ratio = 1.0
                else:
                    actual_fill_ratio = float(
                        rng.uniform(max(0.50, fill_range[0] - 0.10), 0.95)
                    )
                planned_received_qty = min(
                    confirmed_qty,
                    int(np.floor(confirmed_qty * actual_fill_ratio)),
                )
                actual_delivery = promised_date + pd.Timedelta(
                    days=int(rng.integers(*delay_range))
                )
                actual_delivery = max(actual_delivery, po_created + pd.Timedelta(days=1))

                line_quantities: list[int] = []
                if planned_received_qty > 0 and actual_delivery <= DATA_END:
                    if rng.random() < split_probability and planned_received_qty >= 2:
                        first_qty = int(
                            np.clip(
                                round(planned_received_qty * rng.uniform(0.45, 0.75)),
                                1,
                                planned_received_qty - 1,
                            )
                        )
                        line_quantities = [first_qty, planned_received_qty - first_qty]
                    else:
                        line_quantities = [planned_received_qty]

                receipt_dates: list[pd.Timestamp] = []
                cumulative_accepted = 0
                for line_index, received_qty in enumerate(line_quantities, start=1):
                    receipt_date = actual_delivery
                    if line_index > 1:
                        receipt_date += pd.Timedelta(days=int(rng.integers(2, 11)))
                    if receipt_date > DATA_END:
                        continue

                    if rng.random() < 0.015:
                        rejected_qty = received_qty
                    else:
                        rejection_rate = float(
                            rng.choice(
                                [0.0, rng.uniform(0.01, 0.08)], p=[0.88, 0.12]
                            )
                        )
                        rejected_qty = int(np.floor(received_qty * rejection_rate))
                    accepted_qty = received_qty - rejected_qty
                    cumulative_accepted += accepted_qty
                    is_final_complete = cumulative_accepted >= confirmed_qty
                    if accepted_qty == 0 and rejected_qty > 0:
                        receipt_status = "REJECTED"
                    elif len(line_quantities) > 1 and not is_final_complete:
                        receipt_status = "PARTIAL"
                    elif cumulative_accepted < confirmed_qty:
                        receipt_status = "PARTIAL"
                    else:
                        receipt_status = "RECEIVED"

                    receipt_rows.append(
                        {
                            "receipt_id": f"GR{receipt_number:06d}",
                            "receipt_line_id": f"{line_index:03d}",
                            "po_id": po_id,
                            "po_line_id": po_line_id,
                            "receipt_datetime": receipt_date.strftime(
                                "%Y-%m-%d 10:00:00"
                            ),
                            "center_id": center_id,
                            "wms_sku_code": code_lookup[(sku.sku_id, "WMS")],
                            "received_qty": received_qty,
                            "accepted_qty": accepted_qty,
                            "rejected_qty": rejected_qty,
                            "receipt_status": receipt_status,
                        }
                    )
                    receipt_dates.append(receipt_date)
                    receipt_number += 1

                if confirmed_qty == 0:
                    po_status = "Cancelled"
                elif cumulative_accepted >= confirmed_qty:
                    po_status = "Closed"
                elif cumulative_accepted > 0:
                    po_status = "Partial"
                else:
                    po_status = "Open"

                last_updated = (
                    max(receipt_dates) + pd.Timedelta(hours=2)
                    if receipt_dates
                    else min(po_created + pd.Timedelta(days=1), DATA_END)
                )
                po_rows.append(
                    {
                        "po_id": po_id,
                        "po_line_id": po_line_id,
                        "po_created_date": po_created.strftime("%Y-%m-%d"),
                        "vendor_id": sku.default_vendor_id,
                        "erp_sku_code": code_lookup[(sku.sku_id, "ERP")],
                        "center_id": center_id,
                        "ordered_qty": ordered_qty,
                        "confirmed_qty": confirmed_qty,
                        "promised_delivery_date": promised_date.strftime("%Y-%m-%d"),
                        "po_status": po_status,
                        "cancelled_qty": cancelled_qty,
                        "last_updated_at": last_updated.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )
                po_number += 1

    purchase_orders = pd.DataFrame(po_rows, columns=PO_COLUMNS).sort_values(
        ["po_created_date", "po_id"], ignore_index=True
    )
    receipts = pd.DataFrame(receipt_rows, columns=RECEIPT_COLUMNS).sort_values(
        ["receipt_datetime", "receipt_id"], ignore_index=True
    )
    return purchase_orders, receipts


def build_receipt_events(
    receipts: pd.DataFrame,
    mapping: pd.DataFrame,
) -> dict[tuple[str, str], list[tuple[pd.Timestamp, int]]]:
    """Index accepted receipt quantities for weekly inventory simulation."""
    wms_to_sku = dict(
        mapping.loc[mapping["source_system"] == "WMS", ["source_sku_code", "sku_id"]]
        .itertuples(index=False, name=None)
    )
    events: dict[tuple[str, str], list[tuple[pd.Timestamp, int]]] = {}
    for receipt in receipts.itertuples(index=False):
        key = (wms_to_sku[receipt.wms_sku_code], receipt.center_id)
        events.setdefault(key, []).append(
            (pd.Timestamp(receipt.receipt_datetime), int(receipt.accepted_qty))
        )
    return events


def generate_sales_and_inventory(
    sku_master: pd.DataFrame,
    mapping: pd.DataFrame,
    code_lookup: dict[tuple[str, str], str],
    promotions: pd.DataFrame,
    receipts: pd.DataFrame,
    vendor_master: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate inventory-constrained weekly sales and Monday snapshots."""
    promo_lookup = build_promotion_lookup(promotions, mapping)
    receipt_events = build_receipt_events(receipts, mapping)
    vendor_lead_times = vendor_master.set_index("vendor_id")[
        "standard_lead_time_days"
    ].to_dict()
    oms_rows: list[dict[str, object]] = []
    inventory_rows: list[dict[str, object]] = []
    order_number = 1

    for sku in sku_master.itertuples(index=False):
        sigma = VOLATILITY_SIGMA[str(sku.demand_volatility_tier)]
        lead_time_weeks = vendor_lead_times[sku.default_vendor_id] / 7

        for center_id in CENTERS:
            expected_center_demand = float(sku.base_weekly_demand) * CENTER_SHARES[center_id]
            initial_cover = max(3.0, lead_time_weeks + float(rng.uniform(1.0, 3.0)))
            on_hand_qty = int(round(expected_center_demand * initial_cover))
            events = receipt_events.get((sku.sku_id, center_id), [])

            for week_number, week_start in enumerate(WEEK_STARTS, start=1):
                snapshot_time = week_start + pd.Timedelta(hours=8)
                week_end = week_start + pd.Timedelta(days=6, hours=23, minutes=59)

                reserved_qty = min(
                    on_hand_qty, int(np.floor(on_hand_qty * rng.uniform(0.01, 0.08)))
                )
                remaining_after_reserved = on_hand_qty - reserved_qty
                damaged_qty = min(
                    remaining_after_reserved,
                    int(np.floor(on_hand_qty * rng.uniform(0.0, 0.015))),
                )
                quality_hold_qty = min(
                    remaining_after_reserved - damaged_qty,
                    int(np.floor(on_hand_qty * rng.uniform(0.0, 0.025))),
                )
                available_qty = max(
                    0,
                    on_hand_qty - reserved_qty - damaged_qty - quality_hold_qty,
                )

                inventory_rows.append(
                    {
                        "snapshot_datetime": snapshot_time.strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "center_id": center_id,
                        "wms_sku_code": code_lookup[(sku.sku_id, "WMS")],
                        "on_hand_qty": on_hand_qty,
                        "reserved_qty": reserved_qty,
                        "damaged_qty": damaged_qty,
                        "quality_hold_qty": quality_hold_qty,
                        "available_qty": available_qty,
                        "stockout_flag": int(available_qty <= 0),
                    }
                )

                receipts_this_week = sum(
                    quantity
                    for receipt_time, quantity in events
                    if snapshot_time < receipt_time <= week_end
                )
                fulfillable_qty = available_qty + receipts_this_week
                channel_sequence = list(rng.permutation(CHANNELS))
                weekly_fulfilled = 0

                for channel_id in channel_sequence:
                    expected_demand = (
                        float(sku.base_weekly_demand)
                        * CENTER_SHARES[center_id]
                        * CHANNEL_SHARES[channel_id]
                    )
                    seasonality = 1.0 + 0.10 * np.sin(
                        2 * np.pi * (week_number - 1) / 52
                    )
                    demand_noise = float(rng.lognormal(mean=-0.5 * sigma**2, sigma=sigma))
                    promo_uplift, promo_discount = promo_lookup.get(
                        (sku.sku_id, channel_id, week_start), (0.0, 0.0)
                    )
                    demand_mean = max(
                        0.2,
                        expected_demand * seasonality * demand_noise * (1 + promo_uplift),
                    )
                    ordered_qty = int(rng.poisson(demand_mean))
                    if ordered_qty == 0:
                        continue

                    if rng.random() < 0.025:
                        cancelled_qty = ordered_qty
                        fulfilled_qty = 0
                        order_status = "CANCELLED"
                    else:
                        cancelled_qty = int(
                            rng.binomial(ordered_qty, float(rng.uniform(0.0, 0.04)))
                        )
                        net_order_qty = ordered_qty - cancelled_qty
                        fulfilled_qty = min(net_order_qty, fulfillable_qty)
                        fulfillable_qty -= fulfilled_qty
                        weekly_fulfilled += fulfilled_qty
                        if fulfilled_qty == ordered_qty and cancelled_qty == 0:
                            order_status = "COMPLETED"
                        else:
                            order_status = "PARTIAL"

                    oms_rows.append(
                        {
                            "order_id": f"ORD{order_number:08d}",
                            "order_line_id": "001",
                            "order_datetime": min(
                                week_start + pd.Timedelta(days=3, hours=14),
                                pd.Timestamp("2025-12-31 14:00:00"),
                            ).strftime("%Y-%m-%d %H:%M:%S"),
                            "channel_id": channel_id,
                            "fulfillment_center_id": center_id,
                            "oms_sku_code": code_lookup[(sku.sku_id, "OMS")],
                            "ordered_qty": ordered_qty,
                            "fulfilled_qty": fulfilled_qty,
                            "cancelled_qty": cancelled_qty,
                            "unit_selling_price": round(
                                float(sku.list_price) * (1 - promo_discount), 2
                            ),
                            "order_status": order_status,
                        }
                    )
                    order_number += 1

                on_hand_qty = max(
                    0,
                    on_hand_qty + receipts_this_week - weekly_fulfilled,
                )

    sales_orders = pd.DataFrame(oms_rows, columns=OMS_COLUMNS).sort_values(
        ["order_datetime", "order_id"], ignore_index=True
    )
    inventory = pd.DataFrame(inventory_rows, columns=INVENTORY_COLUMNS).sort_values(
        ["snapshot_datetime", "center_id", "wms_sku_code"], ignore_index=True
    )
    return sales_orders, inventory


def generate_manual_overrides(
    sku_master: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Create a small set of human-approved operational exceptions."""
    selected_skus = rng.choice(sku_master["sku_id"], size=12, replace=False)
    reasons = {
        "FORCE_HOLD": "Temporary assortment review",
        "BLOCK_BUY": "Vendor or product compliance hold",
        "EXPEDITE": "Priority replenishment requested",
    }
    rows: list[dict[str, object]] = []

    for index, sku_id in enumerate(selected_skus, start=1):
        override_type = str(
            rng.choice(["FORCE_HOLD", "BLOCK_BUY", "EXPEDITE"], p=[0.3, 0.25, 0.45])
        )
        start_date = WEEK_STARTS[int(rng.integers(4, 48))]
        duration_days = int(rng.choice([7, 14, 21]))
        created_at = start_date - pd.Timedelta(days=int(rng.integers(2, 10)))
        rows.append(
            {
                "override_id": f"OVR{index:04d}",
                "sku_id": sku_id,
                "channel_id": str(rng.choice(CHANNELS)),
                "center_id": str(rng.choice(CENTERS)),
                "effective_from": start_date.strftime("%Y-%m-%d"),
                "effective_to": (start_date + pd.Timedelta(days=duration_days - 1)).strftime(
                    "%Y-%m-%d"
                ),
                "override_type": override_type,
                "override_value": "1",
                "override_reason": reasons[override_type],
                "created_at": created_at.strftime("%Y-%m-%d 09:00:00"),
                "created_by": str(rng.choice(["SCM_PLANNER", "MD_MANAGER", "OPS_LEAD"])),
                "approval_status": str(
                    rng.choice(["APPROVED", "PENDING"], p=[0.9, 0.1])
                ),
            }
        )

    return pd.DataFrame(rows, columns=OVERRIDE_COLUMNS).sort_values(
        ["effective_from", "override_id"], ignore_index=True
    )


def validate_outputs(
    sales_orders: pd.DataFrame,
    inventory: pd.DataFrame,
    purchase_orders: pd.DataFrame,
    receipts: pd.DataFrame,
    promotions: pd.DataFrame,
    overrides: pd.DataFrame,
) -> None:
    """Validate schemas and the main cross-system business constraints."""
    expected_columns = {
        "oms": OMS_COLUMNS,
        "inventory": INVENTORY_COLUMNS,
        "po": PO_COLUMNS,
        "receipt": RECEIPT_COLUMNS,
        "promotion": PROMOTION_COLUMNS,
        "override": OVERRIDE_COLUMNS,
    }
    frames = {
        "oms": sales_orders,
        "inventory": inventory,
        "po": purchase_orders,
        "receipt": receipts,
        "promotion": promotions,
        "override": overrides,
    }
    for name, frame in frames.items():
        if list(frame.columns) != expected_columns[name]:
            raise ValueError(f"Unexpected {name} output schema.")
        if frame.empty:
            raise ValueError(f"{name} output must not be empty.")

    inventory_balance = (
        inventory["on_hand_qty"]
        - inventory["reserved_qty"]
        - inventory["damaged_qty"]
        - inventory["quality_hold_qty"]
    )
    assert inventory["available_qty"].eq(inventory_balance).all()
    assert inventory["stockout_flag"].eq(
        (inventory["available_qty"] <= 0).astype(int)
    ).all()
    assert sales_orders["fulfilled_qty"].le(sales_orders["ordered_qty"]).all()
    assert sales_orders["cancelled_qty"].le(sales_orders["ordered_qty"]).all()
    assert purchase_orders["confirmed_qty"].le(purchase_orders["ordered_qty"]).all()
    assert receipts["received_qty"].eq(
        receipts["accepted_qty"] + receipts["rejected_qty"]
    ).all()
    assert receipts.set_index(["po_id", "po_line_id"]).index.isin(
        purchase_orders.set_index(["po_id", "po_line_id"]).index
    ).all()
    assert len(inventory) == len(WEEK_STARTS) * 40 * len(CENTERS)
    assert (sales_orders["fulfilled_qty"] < sales_orders["ordered_qty"]).any()
    assert (inventory["stockout_flag"] == 1).any()
    assert set(promotions["promo_type"]).issubset(
        {"PRICE_DISCOUNT", "COUPON", "SPECIAL_EXHIBITION"}
    )


def save_csv(dataframe: pd.DataFrame, filename: str) -> Path:
    """Save an output CSV using Excel-friendly UTF-8 encoding."""
    output_path = RAW_DATA_DIR / filename
    dataframe.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def main() -> None:
    """Generate, validate, and save all six transaction source files."""
    rng = np.random.default_rng(RANDOM_SEED)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    sku_master, vendor_master, mapping = load_master_data()
    code_lookup = build_code_lookup(mapping)

    promotions = generate_promotions(sku_master, code_lookup, rng)
    purchase_orders, receipts = generate_purchase_orders_and_receipts(
        sku_master, vendor_master, code_lookup, rng
    )
    sales_orders, inventory = generate_sales_and_inventory(
        sku_master,
        mapping,
        code_lookup,
        promotions,
        receipts,
        vendor_master,
        rng,
    )
    overrides = generate_manual_overrides(sku_master, rng)

    validate_outputs(
        sales_orders,
        inventory,
        purchase_orders,
        receipts,
        promotions,
        overrides,
    )

    outputs = [
        (sales_orders, save_csv(sales_orders, "oms_sales_orders.csv")),
        (inventory, save_csv(inventory, "wms_inventory_snapshot.csv")),
        (receipts, save_csv(receipts, "wms_goods_receipts.csv")),
        (purchase_orders, save_csv(purchase_orders, "erp_purchase_orders.csv")),
        (promotions, save_csv(promotions, "md_promotion_calendar.csv")),
        (overrides, save_csv(overrides, "manual_overrides.csv")),
    ]
    for dataframe, output_path in outputs:
        print(f"Created {output_path} ({len(dataframe)} rows)")


if __name__ == "__main__":
    main()
