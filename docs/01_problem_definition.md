# 01. Problem Definition

## 1. 문서 목적

본 문서는 `Retail InStock Decision MVP`가 해결하려는 비즈니스 문제, 의사결정 단위, 예측 Target, 발주추천 Action 및 MVP 범위를 정의한다. 이후 데이터 소스, Feature, 모델, Rule Engine, Power BI 산출물은 모두 본 문서의 정의를 기준으로 설계한다.

## 2. Business Context

이커머스 리테일에서 관측된 판매량(Observed Sales)은 실제 수요(True Demand)와 동일하지 않다. 판매량에는 다음과 같은 수요·공급 요인이 동시에 반영된다.

- 프로모션 여부와 할인 강도에 따른 일시적 수요 상승
- 품절로 인해 구매 기회가 사라진 검열 수요(Censored Demand)
- 채널 또는 물류센터별 재고 배치 차이
- 예정 대비 입고 지연과 미입고
- 벤더 리드타임 변동 및 납품 이행률
- 최소발주수량(MOQ), 발주 배수, 최소발주금액 등의 발주 제약

따라서 과거 판매량만 단순 연장하여 발주하면 실제 수요를 과소 또는 과대 추정할 수 있으며, 그 결과 결품과 과잉재고가 동시에 발생할 수 있다.

본 MVP는 OMS, WMS, ERP, MD Calendar 및 수기 데이터를 SQLite에 통합하고 SQL Data Mart를 구축한 뒤, 수요예측과 결품 리스크 분류 결과를 Rule Engine에 결합하여 SKU별 실행 가능한 발주추천을 생성하는 의사결정 모델(Decision Model)이다.

## 3. Problem Statement

매주 의사결정 시점(Decision Cutoff) 기준으로 다음 질문에 답한다.

1. 다음 1주 동안 발생할 것으로 예상되는 품절 보정 수요는 얼마인가?
2. 향후 2주 이내에 결품 위험이 있는가?
3. 현재 가용재고, 확정 입고예정, 공급 불확실성 및 발주 제약을 고려할 때 어떤 발주 액션을 취해야 하는가?

예측 자체가 최종 목적은 아니다. 모델 결과를 운영 제약과 연결하여 담당자가 검토하고 실행할 수 있는 발주 수량과 사유를 제공하는 것이 최종 목적이다.

## 4. Decision Grain

분석, Feature 생성, 모델 추론 및 최종 발주추천의 기준 단위는 아래와 같이 고정한다.

> **SKU × 채널(Channel) × 센터(Fulfillment Center) × 주차(Week)**

각 행은 특정 주차의 의사결정 시점에 한 SKU를 특정 채널·센터 조합에서 운영하기 위한 하나의 의사결정 레코드를 의미한다.

- `sku_id`: 내부 표준 SKU 식별자
- `channel_id`: 판매 채널 식별자
- `center_id`: 재고를 보유하고 출고하는 센터 식별자
- `week_start_date`: 주차 기준일

원천 시스템별 상품 코드는 `sku_code_mapping.csv`를 통해 `sku_id`로 표준화한다. 채널이나 센터 수준 정보가 없는 데이터를 임의로 복제하지 않으며, 적용 가능한 Grain으로 명시적으로 배부하거나 집계한다.

## 5. Decision Timeline

주차 `t`의 Feature는 해당 주차의 Decision Cutoff까지 확인 가능한 정보만 사용한다.

| 구분 | 예측 범위 | 정의 |
|---|---:|---|
| Feature window | `t` 및 과거 | Decision Cutoff 시점에 관측·확정된 판매, 재고, 프로모션 계획, PO 계획 및 마스터 정보 |
| Demand target | `t+1` | 다음 주 `stockout_adjusted_sales` |
| Stockout risk target | `t+1`~`t+2` | 향후 2주 내 결품 위험 발생 여부 |
| Decision output | `t` | 현재 시점에서 실행할 발주 수량, 액션 및 사유 |

이 시간축은 학습 데이터와 운영 추론 데이터에 동일하게 적용한다.

## 6. Model Targets

### 6.1 Demand Forecasting Target

- Target name: `target_demand_next_1w`
- Task type: Regression
- Definition: 다음 주의 `stockout_adjusted_sales`

`stockout_adjusted_sales`는 품절로 판매가 제한된 기간의 잠재 수요를 보정한 값이다. 보정 로직은 관측 판매를 실제 수요로 단정하지 않기 위한 장치이며, 학습 시점에는 다음 주의 보정값을 Label로만 사용한다.

### 6.2 Stockout Risk Target

- Target name: `target_stockout_risk_next_2w`
- Task type: Binary Classification
- Definition: 향후 2주 내 결품 위험이 한 번이라도 발생하면 `1`, 아니면 `0`

1차 MVP에서는 운영상 우선순위 선별이 가능하도록 Binary Classification으로 정의한다. 모델의 확률 출력은 Rule Engine에서 위험 임계값과 함께 사용한다.

### 6.3 Reorder Recommendation Outputs

발주추천은 ML Target이 아니라 모델 예측값과 운영 규칙을 결합한 Rule Engine 결과다.

- `recommended_order_qty`: 발주 제약을 반영한 추천 발주수량
- `recommended_action`: 고정된 4개 액션 중 하나
- `action_reason`: 해당 액션을 선택한 주요 근거

`action_reason`은 예측 수요, 재고 커버리지, 결품 위험, 입고 지연, MOQ 및 발주 배수 등 판단 근거를 사람이 검토할 수 있는 형태로 제공한다.

## 7. Reorder Action Definition

액션 라벨은 아래 4개로 고정한다.

| Action | 의미 | 대표 판단 상황 |
|---|---|---|
| `BUY` | 정상 발주 또는 추가 발주 | 예상 수요 대비 재고가 부족하고 정상 리드타임 내 조달 가능한 경우 |
| `HOLD` | 발주 보류 및 현 상태 유지 | 재고와 확정 입고가 목표 재고를 충족하거나 추가 발주 필요성이 낮은 경우 |
| `EXPEDITE` | 기존 PO 긴급 독촉 또는 긴급 조달 | 결품 위험이 높고 정상 발주로는 필요 시점을 맞추기 어려운 경우 |
| `REDUCE` | 발주 축소, 연기 또는 취소 검토 | 과잉재고 위험이 있거나 예정 입고를 포함한 재고가 수요를 크게 초과하는 경우 |

Rule Engine은 모델의 예측을 그대로 액션으로 변환하지 않는다. 재고 상태, 확정 입고, 리드타임, 주문 제약 및 수기 Override를 함께 평가한다. `EXPEDITE`와 `REDUCE`는 신규 발주뿐 아니라 기존 PO에 대한 운영 조치도 포함한다.

## 8. End-to-End Decision Flow

1. 분산된 원천 데이터를 시스템별 CSV로 생성한다.
2. 원천 데이터를 SQLite에 적재하고 코드 및 시간 기준을 정합화한다.
3. SQL로 Decision Grain의 주차 단위 Data Mart를 구축한다.
4. 과거 시점에서 사용 가능했던 정보만으로 Feature와 Target을 생성한다.
5. Regression 모델로 `target_demand_next_1w`를 예측한다.
6. Binary Classification 모델로 `target_stockout_risk_next_2w`의 확률을 산출한다.
7. 예측 결과와 운영 제약을 Rule Engine에 입력하여 발주추천을 생성한다.
8. 의사결정 결과와 설명 변수를 Power BI용 Dataset으로 Export한다.

## 9. MVP Scope

### 9.1 포함 범위 (In Scope)

- Synthetic Source Data
- SQLite DB
- SQL Data Mart
- Demand Forecasting ML
- Stockout Risk Classification ML
- Reorder Decision Rule Engine
- Power BI Dataset Export

### 9.2 제외 범위 (Out of Scope)

- 리뷰 및 검색노출을 핵심 Feature로 사용하는 설계
- 경쟁사 가격 크롤링
- 글로벌 통관 및 HS Code 상세 모델링
- 센터 간 재고이동 최적화
- 강화학습(Reinforcement Learning) 또는 디지털트윈(Digital Twin)

리뷰, 검색노출, 클릭률 및 전환율은 실제 내부 데이터에 접근할 수 있을 때 검토 가능한 확장 Feature이며, 본 MVP의 핵심 Feature와 성능 전제에는 포함하지 않는다.

## 10. Success Criteria

본 MVP의 성공 여부는 모델 정확도만으로 평가하지 않는다.

### 10.1 Data / Pipeline

- 모든 핵심 원천이 표준 `sku_id`와 주차 기준으로 추적 가능해야 한다.
- 동일한 SQL Data Mart에서 학습과 운영 추론용 데이터를 재현할 수 있어야 한다.
- 각 추천 결과가 사용한 Feature 기준일과 원천 데이터로 역추적 가능해야 한다.

### 10.2 Model

- 수요예측 모델은 단순 기준 모델(Baseline) 대비 다음 주 보정 수요 오차를 개선해야 한다.
- 결품 리스크 모델은 클래스 불균형을 고려하여 위험 SKU 선별 성능을 평가해야 한다.
- 평가는 임의 Random Split이 아닌 시간 순서 기반 검증(Time-based Validation)을 우선한다.

### 10.3 Decision

- 모든 레코드에 4개 액션 중 하나와 설명 가능한 `action_reason`이 생성되어야 한다.
- `recommended_order_qty`는 음수가 될 수 없으며 MOQ, 발주 배수 등 제약을 일관되게 반영해야 한다.
- 모델 결과가 누락되거나 데이터 품질이 낮을 때도 보수적인 Fallback Rule이 존재해야 한다.

## 11. Leakage Control Principles

예측 시점에 알 수 없는 정보는 Feature로 사용하지 않는다. 특히 아래 항목을 명시적으로 금지한다.

- 미래 실제 판매량(Future Actual Sales)
- 미래 실제 입고일 또는 실제 입고수량(Future Actual Receipt)
- 미래 실제 결품 여부(Future Actual Stockout)
- 현재 프로모션 종료 후 계산되는 미래 실제 Uplift

미래 값은 Target 산출과 사후 평가에만 사용할 수 있다. 예정 정보는 Decision Cutoff 시점에 실제로 확정되어 있던 계획값과 당시 상태만 사용하며, 이후 수정된 최신 값으로 과거 레코드를 덮어쓰지 않는다.

## 12. 주요 가정 및 한계

- 본 MVP의 데이터는 실제 운영 구조를 모사한 Synthetic Data이며 실제 기업의 정책을 대표하지 않는다.
- 품절 보정 수요는 관측 불가능한 잠재 수요의 추정치이므로 보정 방법에 따른 불확실성이 존재한다.
- 추천 액션은 담당자의 의사결정을 지원하는 Decision Support 결과이며 자동 발주 확정이 아니다.
- 비용, 서비스레벨, 위험 임계값은 MVP 가정값으로 시작하고 실제 운영 적용 시 조직 정책에 맞게 보정해야 한다.
