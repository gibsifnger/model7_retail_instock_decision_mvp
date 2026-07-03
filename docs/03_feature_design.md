# 03. Feature Design

## 1. 문서 목적

본 문서는 수요예측(Demand Forecasting), 결품 리스크 분류(Stockout Risk Classification), 발주 Rule Engine에 사용할 Feature의 정의와 생성 원칙을 설명한다. 모든 Feature는 **SKU × 채널 × 센터 × 주차** Grain과 Decision Cutoff를 기준으로 생성한다.

## 2. Feature Design Principles

### 2.1 Point-in-Time Availability

주차 `t`의 Feature는 `t`의 Decision Cutoff까지 실제로 알 수 있었던 정보만 사용한다. 판매·재고·입고 실적은 Cutoff 이전 관측값만 포함하고, 프로모션과 PO는 당시 확정된 계획값만 포함한다.

### 2.2 Grain Consistency

모든 Feature는 최종 Decision Grain으로 집계한다. 원천 Grain이 더 거친 경우 무조건 반복 복제하지 않으며, 명시적인 배부 규칙 또는 적용 범위 Flag를 사용한다.

### 2.3 Interpretability

Rule Engine과 운영 담당자가 결과를 해석할 수 있도록 수요 추세, 재고 커버리지, 공급 변동성, 주문 제약 등 비즈니스 의미가 분명한 Feature를 우선한다.

### 2.4 Missingness

Null은 정보 부재를 의미할 수 있으므로 단순히 0으로 대체하지 않는다. 적절한 대체값과 함께 `*_missing_flag`, 데이터 이력 길이 등 보조 Flag를 유지한다.

## 3. Target Definitions

Target은 Feature가 아니며 학습 Label로만 사용한다.

| Target | Task | 정의 |
|---|---|---|
| `target_demand_next_1w` | Regression | 동일 Grain의 다음 주 `stockout_adjusted_sales` |
| `target_stockout_risk_next_2w` | Binary Classification | 동일 Grain에서 향후 2주 내 결품 위험이 있으면 `1`, 아니면 `0` |

발주추천 결과인 `recommended_order_qty`, `recommended_action`, `action_reason`은 ML Target이 아니다. 예측값과 재고·공급·제약 Feature를 입력받는 Rule Engine Output이다.

## 4. Feature Groups

### 4.1 Sales Features

판매 수준, 추세, 계절성 및 변동성을 표현한다. 품절에 의해 판매가 제한된 구간은 관측 판매와 보정 판매를 구분한다.

| Feature | 정의 | 목적 |
|---|---|---|
| `sales_qty_1w` | 최근 완료 1주 유효 판매수량 | 최신 판매 수준 |
| `sales_lag_1w` | 직전 1주 판매수량 | 단기 자기상관 |
| `sales_lag_2w` | 2주 전 판매수량 | 단기 추세 비교 |
| `sales_lag_4w` | 4주 전 판매수량 | 월간 반복 패턴 |
| `sales_lag_52w` | 전년 동주 판매수량, 이력 존재 시 | 연간 계절성 |
| `sales_rolling_mean_4w` | 직전 4주 평균 판매량 | 단기 기준 수요 |
| `sales_rolling_mean_8w` | 직전 8주 평균 판매량 | 중기 기준 수요 |
| `sales_rolling_std_4w` | 직전 4주 판매 표준편차 | 단기 변동성 |
| `sales_trend_4w` | 최근 4주 판매 추세 기울기 또는 전·후반 평균 차이 | 상승·하락 추세 |
| `demand_volatility_index` | 평균 대비 판매 변동성 | 예측 불확실성 |
| `nonzero_sales_weeks_8w` | 최근 8주 중 판매가 발생한 주 수 | 간헐 수요 구분 |
| `stockout_adjusted_sales_lag_1w` | 직전 주 품절 보정 판매량 | 검열 수요 보정 이력 |
| `stockout_adjusted_sales_rolling_mean_4w` | 직전 4주 품절 보정 판매 평균 | 보정 기준 수요 |

Rolling 및 Lag Feature에는 현재 주의 미완료 실적이나 미래 주 판매가 포함되지 않는다.

### 4.2 Demand Distortion Features

관측 판매가 실제 수요를 충분히 반영하지 못하는 정도를 표현한다.

| Feature | 정의 | 목적 |
|---|---|---|
| `stockout_days_last_1w` | 직전 1주 품절 일수 | 최근 판매 검열 정도 |
| `stockout_days_last_4w` | 직전 4주 누적 품절 일수 | 반복 품절 영향 |
| `sales_censored_flag` | 판매 가능 시간이 부족하여 판매가 검열된 주 여부 | 관측 판매 신뢰도 |
| `in_stock_rate_4w` | 최근 4주 중 판매 가능 시간 비율 | 수요 관측 가능성 |
| `lost_sales_estimate_4w` | 최근 4주의 추정 미실현 판매량 | 잠재 수요 규모 |
| `adjustment_ratio_4w` | 보정 판매량 / 관측 판매량 | 보정 강도 |
| `center_allocation_shortage_flag` | 해당 센터 배치 부족으로 판매 기회가 제한된 여부 | 재고 위치 왜곡 |
| `new_product_flag` | 출시 후 초기 구간 여부 | 짧은 이력 구분 |
| `sales_history_weeks` | 사용 가능한 유효 판매 이력 주 수 | Feature 신뢰도 |

`stockout_adjusted_sales`는 Target 정의와 과거 Lag 생성의 기반이지만, 동일 주의 미래까지 반영한 보정값을 현재 Feature에 사용하지 않는다.

### 4.3 Promotion Features

Decision Cutoff 시점에 알려진 프로모션 계획과 과거 종료된 프로모션의 반응을 표현한다.

| Feature | 정의 | 목적 |
|---|---|---|
| `promo_flag` | 대상 주 프로모션 진행 또는 확정 계획 여부 | 행사 수요 구분 |
| `promo_type` | 가격할인, 쿠폰, 기획전 등 행사 유형 | 반응 차이 학습 |
| `promo_depth` | 정상가 대비 계획 할인율 | 프로모션 강도 |
| `promo_price_index` | 계획 프로모션가 / 정상가 | 가격 수준 |
| `promo_duration_days` | 전체 계획 행사 기간 | 행사 구조 |
| `promo_days_in_week` | 해당 주에 포함되는 행사 일수 | 주간 노출 강도 |
| `promo_day_index` | 행사 시작 후 해당 주의 상대 위치 | 초반·후반 효과 |
| `weeks_to_promo_start` | 확정된 다음 행사 시작까지 남은 주 | 사전 구매·발주 대응 |
| `historical_promo_uplift` | Cutoff 이전에 종료된 유사 행사의 평균 Uplift | 과거 행사 반응 |
| `promo_overlap_count` | 동일 기간 중첩 행사 수 | 행사 중첩 영향 |

`historical_promo_uplift`는 해당 SKU의 이력이 부족하면 카테고리·프로모션 유형 수준으로 Backoff하고 적용 수준을 별도 Flag로 남긴다. 현재 프로모션이 끝난 뒤 확인되는 실제 Uplift는 현재 또는 과거 예측 Feature에 사용할 수 없다.

### 4.4 Inventory Features

센터의 현재 판매 가능 재고, 입고예정 및 예상 수요 대비 재고 충분도를 표현한다.

| Feature | 정의 | 목적 |
|---|---|---|
| `on_hand_qty` | Cutoff 시점 장부상 보유재고 | 물리 재고 수준 |
| `reserved_qty` | 이미 할당된 재고 | 판매 가능 수량 보정 |
| `available_qty` | 즉시 판매 가능한 재고 | 핵심 공급 가용성 |
| `quality_hold_qty` | 품질보류 재고 | 비가용 재고 위험 |
| `inbound_qty_next_1w` | Cutoff 당시 확정된 1주 내 PO 잔량 | 단기 예정 공급 |
| `inbound_qty_next_2w` | Cutoff 당시 확정된 2주 내 PO 잔량 | 위험 Horizon 공급 |
| `inbound_qty_next_4w` | Cutoff 당시 확정된 4주 내 PO 잔량 | 중기 공급 |
| `inventory_position_qty` | 가용재고 + 확정 입고 - 미충족 할당량 | 발주 기준 재고 |
| `inventory_cover_weeks` | 재고 Position / 기준 주간 수요 | 재고 충분도 |
| `safety_stock_qty` | 서비스 수준 및 변동성을 반영한 안전재고 | 목표 재고 하한 |
| `days_since_last_stockout` | 마지막 관측 품절 이후 경과일 | 최근 품절성 |
| `stockout_frequency_8w` | 최근 8주 품절 발생 주 비율 | 반복 결품 위험 |
| `excess_inventory_flag` | 재고 커버리지가 상한을 초과하는지 여부 | `REDUCE` 판단 |

`inventory_cover_weeks`의 분모가 0 또는 지나치게 작을 때는 상한 처리하고 별도 Flag를 둔다. 향후 실제 입고는 포함하지 않으며 Cutoff 당시의 확정 PO 계획만 포함한다.

### 4.5 Vendor / Supply Features

발주와 실제 입고 이력으로 벤더의 속도, 변동성 및 이행 신뢰도를 표현한다.

| Feature | 정의 | 목적 |
|---|---|---|
| `standard_lead_time_days` | Vendor Master의 계약 리드타임 | 기준 조달 기간 |
| `vendor_avg_lead_time` | Cutoff 이전 완료 PO의 실제 평균 리드타임 | 경험적 조달 기간 |
| `vendor_lead_time_std` | 과거 실제 리드타임 표준편차 | 공급 불확실성 |
| `lead_time_p90_days` | 과거 리드타임 90백분위 | 보수적 공급 기간 |
| `po_fill_rate` | 과거 발주수량 대비 유효 입고수량 | 수량 이행률 |
| `on_time_delivery_rate` | 약속 납기 내 완료된 PO 비율 | 납기 신뢰도 |
| `open_po_qty` | Cutoff 시점 미입고 PO 잔량 | 예정 공급 규모 |
| `overdue_po_qty` | Cutoff 당시 약속 납기를 넘긴 PO 잔량 | 즉시 공급 위험 |
| `open_po_count` | 진행 중 PO Line 수 | 공급 파이프라인 복잡도 |
| `days_to_next_promised_receipt` | 가장 가까운 당시 약속 납기까지 일수 | 결품 전 입고 가능성 |
| `supplier_risk_score` | 이행률·변동성을 조합한 과거 기반 점수 | 공급 리스크 요약 |

실제 리드타임 통계는 Cutoff 이전에 입고 완료된 PO만 사용한다. 아직 입고되지 않은 PO의 미래 실제 완료일을 사용해 과거 통계를 계산하지 않는다.

### 4.6 Ordering Constraint Features

추천수량의 실행 가능성과 Rule Engine의 반올림·차단 조건을 표현한다.

| Feature | 정의 | 목적 |
|---|---|---|
| `moq_qty` | 최소발주수량 | 최소 주문 제약 |
| `order_multiple` | 허용 발주 배수 | 수량 반올림 |
| `min_order_amount` | 최소발주금액 | 금액 제약 |
| `unit_cost` | 발주 단위 원가 | 발주금액 계산 |
| `order_cycle_days` | 정기 발주 주기 | Review Period 반영 |
| `days_to_next_order_cycle` | 다음 발주 가능일까지 일수 | 주문 시점 판단 |
| `max_order_qty` | 정책상 최대 발주수량, 적용 시 | 과대 발주 방지 |
| `case_pack_qty` | 박스 또는 Case 단위 수량 | 물류 단위 정합성 |
| `order_block_flag` | 거래중지·단종 등 발주 차단 여부 | 강제 `HOLD`/`REDUCE` |
| `manual_override_flag` | 승인된 수기 예외 존재 여부 | 최종 규칙 우선순위 |

발주 제약은 모델 Feature로 활용할 수 있으나, 최종 `recommended_order_qty` 산출 시 Rule Engine에서 반드시 다시 적용한다.

### 4.7 Product Features

SKU의 구조적 특성과 수명주기 정보를 표현한다.

| Feature | 정의 | 목적 |
|---|---|---|
| `category_l1`, `category_l2` | 상품 계층 | 유사 상품 패턴 학습 |
| `brand` | 브랜드 | 브랜드별 수요 차이 |
| `unit_cost` | 단위 원가 | 재고·발주 비용 맥락 |
| `list_price` | 정상 판매가 | 가격대 구분 |
| `shelf_life_days` | 사용 가능 기간 | 과잉재고 위험 |
| `product_age_weeks` | 출시 후 경과 주 수 | Lifecycle 반영 |
| `new_product_flag` | 신상품 기준 기간 해당 여부 | Cold Start 구분 |
| `discontinue_soon_flag` | 단종 예정 임박 여부 | `REDUCE`/발주 차단 |
| `active_flag` | 현재 운영 SKU 여부 | 대상 필터링 |

범주형 Feature는 학습 데이터에서 일관된 Encoding을 적용하고, 미등록 신규 범주를 처리할 수 있어야 한다.

## 5. Feature Use by Component

| Feature group | Demand Forecast | Stockout Risk | Rule Engine |
|---|:---:|:---:|:---:|
| Sales | 핵심 | 핵심 | 참고 |
| Demand Distortion | 핵심 | 핵심 | 설명 |
| Promotion | 핵심 | 보조 | 설명 |
| Inventory | 핵심 | 핵심 | 핵심 |
| Vendor / Supply | 보조 | 핵심 | 핵심 |
| Ordering Constraint | 보조 | 보조 | 핵심 |
| Product | 핵심 | 보조 | 제약·설명 |

실제 Feature 선택은 시간 기반 검증 결과와 데이터 가용성에 따라 축소할 수 있으나, 핵심 Feature 범위를 리뷰·검색 지표로 대체하지 않는다.

## 6. Reorder Rule Engine Inputs and Outputs

Rule Engine은 다음 입력을 결합한다.

- Demand Forecast의 다음 주 예측값
- Stockout Classifier의 향후 2주 위험 확률
- `available_qty`, `inventory_position_qty`, 확정 입고예정
- 안전재고 및 목표 Cover
- 벤더 리드타임과 공급 신뢰도
- MOQ, 발주 배수, 최소발주금액 및 발주 차단 조건
- 승인된 Manual Override

개념적인 필요수량은 다음 관계를 따른다.

```text
raw_order_need
= target_inventory_qty
- available_qty
- eligible_confirmed_inbound_qty
```

이후 음수 하한, MOQ, 발주 배수, 최소발주금액, 단종 및 Override를 순서대로 적용하여 `recommended_order_qty`를 산출한다. 정확한 임계값과 우선순위는 별도의 Rule 설정으로 관리한다.

액션의 대표적인 방향은 다음과 같다.

- `BUY`: 필요수량이 있고 정상 리드타임 내 공급 가능한 경우
- `HOLD`: 재고 Position이 충분하거나 발주 필요가 없는 경우
- `EXPEDITE`: 2주 결품 위험이 높고 기존 정상 공급이 필요 시점보다 늦는 경우
- `REDUCE`: 과잉 Cover, 수요 하락, 단종 임박 등으로 기존 또는 신규 발주 축소가 필요한 경우

## 7. Leakage Control

### 7.1 금지 Feature

아래 정보는 예측 시점에 알 수 없으므로 Feature로 사용하지 않는다.

- 미래 실제 판매량
- 미래 실제 입고일 및 미래 실제 입고수량
- 미래 실제 결품 여부
- 현재 프로모션의 종료 후 계산된 미래 실제 Uplift

### 7.2 허용 범위

- 과거 완료 주차의 판매, 재고, 품절 및 입고 실적
- Decision Cutoff 당시 확정된 프로모션 계획
- Decision Cutoff 당시 존재한 Open PO와 당시 약속 납기
- Cutoff 이전 종료된 프로모션에서 계산한 Historical Uplift
- Cutoff 이전 완료된 PO에서 계산한 Vendor Performance

### 7.3 구현 통제

- Lag와 Rolling Window는 현재 행 이전 시점에서 종료한다.
- Feature Query와 Target Query를 분리한다.
- PO와 프로모션은 최신 상태가 아닌 As-of Version으로 과거 시점 Join을 수행한다.
- 전처리, 결측치 대체 및 Encoding은 Train 구간에서 Fit하고 Validation/Test에 적용한다.
- 데이터 분할은 시간 순서를 보존하며 동일한 미래 정보가 Train으로 역류하지 않도록 한다.

## 8. Optional Extension Features

아래 Feature는 실제 내부 데이터에 접근할 수 있을 때 확장 후보로만 고려한다.

- 리뷰 수, 평점, 리뷰 감성
- 검색노출 및 검색 순위
- 클릭률(CTR)
- 전환율(CVR)

이 지표들은 수요 신호를 보강할 가능성이 있지만 수집 정의, 지연 시간, 채널별 일관성 및 Leakage 검증이 필요하다. 따라서 본 MVP의 핵심 Feature, 필수 입력 또는 성능 가정에는 포함하지 않는다.

## 9. Feature Validation Checklist

- 각 Feature가 Decision Grain에서 유일한가?
- Feature 생성에 사용한 모든 원천 레코드가 Decision Cutoff 이전에 사용 가능했는가?
- Rolling Window와 Lag의 시간 방향이 올바른가?
- 판매 0과 데이터 미수집을 구분했는가?
- 품절 구간의 판매를 정상 수요 0으로 해석하지 않았는가?
- 센터 공용 재고를 채널별로 중복 계산하지 않았는가?
- Open PO에서 취소·기입고 수량을 차감했는가?
- 완료되지 않은 프로모션의 실제 Uplift를 사용하지 않았는가?
- 모델 입력과 Rule Engine 입력의 단위가 일치하는가?
- 산출 Feature의 정의, 기준일, Source 및 Version을 추적할 수 있는가?

## 10. MVP v1 Feature Implementation Scope

본 문서의 전체 Feature 후보는 중장기 설계 범위로 유지한다. 다만 1차 MVP에서 실제 구현하는 Core Feature는 아래 목록으로 고정하며, 데이터마트·Feature Engineering·모델 학습 및 추론 구현 시 우선 적용한다.

### 10.1 Sales

- `sales_lag_1w`
- `sales_lag_4w`
- `sales_rolling_mean_4w`
- `sales_rolling_std_4w`
- `demand_volatility_index`
- `stockout_adjusted_sales_rolling_mean_4w`

### 10.2 Demand Distortion

- `stockout_days_last_4w`
- `sales_censored_flag`
- `in_stock_rate_4w`
- `sales_history_weeks`

### 10.3 Promotion

- `promo_flag`
- `promo_type`
- `promo_depth`
- `promo_duration_days`
- `promo_days_in_week`
- `promo_day_index`
- `historical_promo_uplift`

### 10.4 Inventory

- `available_qty`
- `reserved_qty`
- `inbound_qty_next_1w`
- `inbound_qty_next_2w`
- `inbound_qty_next_4w`
- `inventory_position_qty`
- `inventory_cover_weeks`
- `safety_stock_qty`

### 10.5 Vendor / Supply

- `standard_lead_time_days`
- `vendor_avg_lead_time`
- `vendor_lead_time_std`
- `po_fill_rate`
- `on_time_delivery_rate`
- `open_po_qty`
- `overdue_po_qty`

### 10.6 Ordering Constraint

- `moq_qty`
- `order_multiple`
- `min_order_amount`
- `unit_cost`
- `order_block_flag`
- `manual_override_flag`

### 10.7 Product

- `category_l1`
- `category_l2`
- `brand`
- `list_price`
- `product_age_weeks`
- `new_product_flag`
- `active_flag`

### 10.8 Implementation Priority and Extension Policy

- 위 목록에 포함되지 않은 나머지 Feature는 v2 확장 후보로 관리한다.
- Codex 구현 시에는 MVP v1 Core Feature를 우선 구현한다.
- 향후 Feature가 추가되더라도 **SKU × 채널 × 센터 × 주차** Decision Grain은 변경하지 않는다.
- 향후 Feature가 추가되더라도 본 문서의 Point-in-Time Availability 및 Leakage Control 원칙은 동일하게 적용한다.
