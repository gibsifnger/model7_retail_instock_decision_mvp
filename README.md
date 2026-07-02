# Retail InStock Decision MVP

본 프로젝트는 쿠팡·올리브영과 같은 이커머스 리테일 환경에서 발생하는 수요예측, 결품 리스크, 재고 건전성, 발주추천 문제를 하나의 의사결정 흐름으로 연결한 SCM 데이터 분석 MVP입니다.

단일 CSV 분석이 아니라, OMS/WMS/ERP/MD Calendar에 흩어진 데이터를 SQL 기반 데이터마트로 통합하고, Python 머신러닝 모델과 Power BI 대시보드까지 연결하는 구조로 설계했습니다.

## 핵심 문제

이커머스 리테일 환경의 판매량은 실제 수요 그 자체가 아닙니다.

프로모션, 품절, 가격 변화, 신상품 효과, 리드타임 지연, 벤더 공급 안정성 등이 섞인 관측값입니다.

따라서 단순 과거 판매량 기반 발주는 결품과 과잉재고를 동시에 악화시킬 수 있습니다.

## 프로젝트 목표

- 분산된 SCM 원천 데이터를 SQL로 통합
- SKU × 채널 × 센터 × 주차 단위 분석 데이터마트 구축
- 머신러닝 기반 수요예측 모델 설계
- 결품 리스크 분류 모델 설계
- 발주추천 Rule Engine 설계
- Power BI 운영 대시보드용 데이터셋 생성

## Pipeline

OMS / WMS / ERP / MD Calendar / Manual Data
        ↓
SQL Data Mart
        ↓
Feature Engineering
        ↓
Demand Forecast ML
        ↓
Stockout Risk Classifier
        ↓
Reorder Decision Engine
        ↓
Power BI Dashboard

## Main Features

### Demand Features

- sales_lag_1w
- sales_lag_4w
- sales_rolling_mean_4w
- sales_rolling_std_4w
- demand_volatility_index

### Promotion Features

- promo_flag
- promo_depth
- promo_type
- promo_day_index
- promo_duration_days
- historical_promo_uplift

### Stockout Correction Features

- stockout_days_last_4w
- sales_censored_flag
- stockout_adjusted_sales

### Inventory Features

- available_qty
- inbound_qty_next_4w
- inventory_cover_weeks
- safety_stock_qty

### Vendor / Supply Features

- vendor_avg_lead_time
- vendor_lead_time_std
- po_fill_rate
- on_time_delivery_rate

### Ordering Constraint Features

- moq_qty
- order_multiple
- min_order_amount

## Tech Stack

- Python
- Pandas
- Scikit-learn
- SQLite
- SQL
- Power BI

## Project Status

현재 단계는 초기 프로젝트 구조 설계 단계입니다.

1. Synthetic Source Data 생성
2. SQLite DB 구축
3. SQL Data Mart 생성
4. Feature Engineering
5. Demand Forecasting Model 학습
6. Stockout Risk Classifier 학습
7. Reorder Recommendation 생성
8. Power BI Dataset Export
