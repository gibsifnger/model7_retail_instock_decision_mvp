# Retail InStock Decision MVP

> SKU 수요예측 기반 결품 리스크·발주추천 의사결정 모델

이 프로젝트는 **synthetic data 기반 포트폴리오용 MVP**입니다. 실제 회사 데이터를 사용하지 않았으며, 이커머스 리테일의 재고·발주 의사결정 흐름을 재현하기 위해 가상의 OMS/WMS/ERP/MD Calendar/Manual Override 데이터를 생성했습니다.

단순한 수요예측 모델이 아니라, 분산된 운영 데이터를 통합해 **SKU × Channel × Center × Week** 단위로 수요예측(Demand Forecast), 결품 리스크 분류(Stockout Risk Classification), 발주추천 액션(Reorder Recommendation)을 생성하는 **Retail InStock 의사결정 파이프라인**입니다.

## 1. 프로젝트 목적

이커머스 리테일에서 관측 판매량(observed sales)은 실제 수요(true demand)가 아닙니다. 프로모션, 품절, 재고 위치, 입고 지연, 벤더 리드타임, MOQ/발주배수 같은 운영 제약이 섞인 결과입니다.

본 프로젝트는 OMS/WMS/ERP/MD Calendar/Manual Override 데이터를 통합하여 **SKU × 채널 × 센터 × 주차** 단위로 다음 의사결정을 지원합니다.

- 다음 주 수요예측(Demand Forecast)
- 향후 2주 결품 리스크 분류(Stockout Risk Classification)
- 발주추천 액션(Reorder Rule Engine)
- KPI Summary 및 Power BI Dataset Export

## 2. Decision Grain

| 항목 | 기준 |
|---|---|
| Decision Grain | SKU × Channel × Center × Week |
| Time Unit | 주차(week_start_date, Monday 기준) |
| Demand Target | target_demand_next_1w |
| Stockout Target | target_stockout_risk_next_2w |
| Reorder Output | recommended_order_qty, recommended_action, action_reason |

## 3. End-to-End Pipeline

```text
Synthetic Source Data
→ SQLite DB
→ SQL Mart
→ Final Feature Table
→ Demand Forecast Model
→ Stockout Risk Classifier
→ Reorder Rule Engine
→ KPI Summary
→ Power BI Dataset Export
```

## 4. 데이터 소스 구조

| Source File | Source System | 역할 |
|---|---|---|
| `sku_master.csv` | Product Master | SKU 카테고리, 브랜드, 가격, 원가, MOQ, 발주배수, 상품 상태 기준정보 |
| `vendor_master.csv` | Vendor Master | 벤더 국가, 수입 여부, 기준 리드타임, 주문 주기, 결제조건, synthetic 공급 안정성 seed |
| `sku_code_mapping.csv` | Code Mapping | OMS/WMS/ERP/MD 시스템별 SKU 코드 매핑 |
| `oms_sales_orders.csv` | OMS | 주문수량, 출고수량, 취소수량, 판매가격, 주문 상태 |
| `wms_inventory_snapshot.csv` | WMS | 센터별 재고 스냅샷, 가용재고, 예약재고, 손상/보류재고, 결품 여부 |
| `erp_purchase_orders.csv` | ERP | PO 생성일, 벤더, 주문수량, 확정수량, 약속 납기일, PO 상태 |
| `wms_goods_receipts.csv` | WMS | PO와 연결된 실제 입고일, 입고수량, 승인/반려수량 |
| `md_promotion_calendar.csv` | MD Calendar | 프로모션 유형, 기간, 할인율, 프로모션 가격 |
| `manual_overrides.csv` | Manual | FORCE_HOLD, BLOCK_BUY, EXPEDITE 등 수동 운영 예외 규칙 |

## 5. 주요 산출물

| Output | 설명 |
|---|---|
| `data/mart/final_modeling_table.csv` | 모델링용 최종 Feature Table |
| `outputs/predictions/demand_forecast_result.csv` | 수요예측 결과 및 baseline 비교 |
| `outputs/predictions/stockout_risk_result.csv` | 결품 리스크 예측 확률, 예측 라벨, baseline 비교 |
| `outputs/decisions/reorder_action_result.csv` | SKU × Channel × Center × Week 단위 발주추천 액션 |
| `outputs/metrics/kpi_summary.csv` | 포트폴리오/대시보드용 KPI 요약 테이블 |
| `data/powerbi/powerbi_overview_kpi.csv` | Power BI Overview KPI dataset |
| `data/powerbi/powerbi_reorder_action.csv` | Power BI Reorder Action dataset |

## 6. 모델 성능 요약

### Demand Forecast

| Metric | Model | Baseline |
|---|---:|---:|
| WAPE | 0.3289 | 0.3628 |
| Bias | -0.0058 | -0.1016 |
| Forecast Accuracy | 0.6711 | 0.6372 |
| WAPE Improvement | 3.39%p | - |

### Stockout Risk

| Metric | Model | Baseline |
|---|---:|---:|
| Precision | 0.7007 | 0.2810 |
| Recall | 0.7464 | 0.4928 |
| F1 | 0.7228 | 0.3579 |
| PR-AUC | 0.8220 | 0.1628 |
| ROC-AUC | 0.9841 | - |
| False Negative | 35 | 70 |

> Stockout Risk 모델은 운영 기준 threshold 0.4 결과를 대표값으로 사용했습니다.

## 7. Reorder Decision 결과

| Metric | Value |
|---|---:|
| Total Decision Rows | 2,520 |
| Total Recommended Order Qty | 5,454 |
| HOLD | 1,650 / 65.5% |
| REDUCE | 796 / 31.6% |
| BUY | 50 / 2.0% |
| EXPEDITE | 24 / 1.0% |

발주추천은 ML Target이 아니라 Rule Engine Output입니다. 모델 예측값을 그대로 발주 확정으로 사용하지 않고, MOQ, 발주배수, 최소주문금액, 재고커버, 결품 리스크, 오픈 PO, 수동 override를 반영한 **Decision Support** 결과로 생성합니다.

## 8. 실행 방법

가상환경 활성화 후 전체 파이프라인을 한 번에 실행합니다.

```bash
python run_pipeline.py
```

개별 단계 실행 순서는 아래와 같습니다.

```bash
python src/data_generation/03_generate_all_raw_data.py
python src/database/build_sqlite_db.py
python src/database/run_sql_scripts.py
python src/features/build_final_feature_table.py
python src/models/train_demand_forecast_model.py
python src/models/train_stockout_classifier.py
python src/decision_engine/reorder_rule_engine.py
python src/reporting/make_kpi_summary.py
python src/reporting/export_powerbi_dataset.py
```

## 9. 포트폴리오 해석 포인트

- 단일 CSV 예측이 아니라 OMS/WMS/ERP/MD 데이터를 SQL로 통합한 구조입니다.
- 수요예측(Demand Forecast)과 결품 리스크(Stockout Risk)를 분리했습니다.
- 발주추천은 ML Target이 아니라 Rule Engine Output으로 설계했습니다.
- 모델 결과는 발주 확정이 아니라 운영 의사결정을 돕는 Decision Support로 사용합니다.
- Leakage 방지를 위해 Feature와 Target을 분리하고 Time-based Validation을 사용했습니다.
- Power BI Dataset까지 Export하여 대시보드 연결이 가능한 형태로 마무리했습니다.

## 10. 한계와 확장 방향

본 프로젝트의 데이터는 synthetic data입니다. 실제 운영 적용을 위해서는 ERP/WMS/OMS 실제 이력, 프로모션 캘린더, 발주 승인 정책과 연결해야 합니다.

향후 확장 후보는 다음과 같습니다.

- 실제 판매/재고 이력 연동
- 제품별 서비스레벨 정책(Service Level Policy)
- 재고이동 최적화(Inventory Rebalancing Optimization)
- Power BI 대시보드 고도화
- Agentic SCM 모니터링
- Review/Search/CTR/CVR 등 수요 신호 확장

## 11. Tech Stack

| 영역 | 기술 |
|---|---|
| Data Processing | Python, pandas, numpy |
| Database | SQLite |
| SQL Mart | SQL |
| Modeling | scikit-learn |
| Reporting | CSV Export, Power BI Dataset |
| Orchestration | Python subprocess |
