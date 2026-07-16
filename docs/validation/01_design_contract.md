# Model7 Design Contract

## 1. Business Decision

- Model7이 내리려는 의사결정:
  - 채널별 주간 수요와 향후 결품 위험을 예측한다.
  - 센터 재고, 입고 예정, Open PO, 리드타임, MOQ, 발주배수를 반영한다.
  - 최종적으로 BUY / HOLD / EXPEDITE / REDUCE 액션과 추천수량을 생성한다.

- 의사결정을 사용하는 사람:
  - InStock 담당자
  - SCM Planner
  - 재고·보충발주 담당자
  - 센터 운영 및 구매 담당자

- 의사결정 결과:
  - 다음 1주 수요예측
  - 향후 2주 결품위험
  - 추천 액션
  - 추천 발주수량
  - 액션 사유

## 2. Decision Time / As-of

- 현재 구현 추정:
  - 각 행의 기준일은 월요일인 week_start_date이다.
  - WMS 재고와 PO Position은 월요일 오전 08:00 기준으로 계산된다.
  - 판매 데이터는 해당 week_start_date가 속한 주간 전체 실적을 집계한다.
  - 판매 Lag와 Rolling Feature는 shift(1)을 적용하여 이전 완료 주차만 사용한다.
  - 일부 현재 주차 정보가 모델 Feature로 직접 사용되는지는 추가 검증이 필요하다.

- 최종 확정:
  - PENDING

- 판단 기준시각:
  - 후보안: 매주 월요일 오전 08:00
  - 최종 확정 전 Feature별 정보 가용시점 검증 필요

- 사용 가능한 데이터:
  - 이전 주 일요일까지 확정된 판매이력
  - 월요일 오전 08:00 WMS 가용재고
  - 월요일 오전 08:00 기준 Open PO 및 입고실적
  - 판단시점 이전에 확정된 프로모션 계획
  - 판단시점 이전에 완료된 공급사 납기이력

- 사용하면 안 되는 데이터:
  - 판단시점 이후 발생한 실제 판매
  - 판단시점 이후 확정된 입고결과
  - 미래 실제 결품 여부
  - 판단시점 이후 등록·승인된 프로모션 및 Override

## 3. Data Grain

| 데이터 | 현재 Grain | 목표 Grain | 검증 필요사항 |
|---|---|---|---|
| OMS Sales Raw | 주문번호 × 주문라인 | 원천 유지 | 주문수량·출고수량·취소수량 의미 |
| OMS Sales Weekly | SKU × 채널 × 센터 × 주차 | 동일 | Mapping·주간 집계 수량 보존 |
| WMS Inventory | SKU × 센터 × 주차 | 동일 | 물리 재고는 채널 공용인지 확인 |
| ERP PO Position | SKU × 센터 × 주차 | 동일 | 주차별 As-of 잔량 계산 확인 |
| Goods Receipt Raw | 입고번호 × 입고라인 | 원천 유지 | PO와 입고수량 대사 |
| Promotion | SKU × 채널 × 주차 | 동일 | 중복 프로모션 선택 기준 |
| SQL Mart | SKU × 채널 × 센터 × 주차 | 분석용 Mart로 유지 검토 | 센터 재고·PO가 채널별 반복됨 |
| Demand Prediction | SKU × 채널 × 센터 × 주차 | 동일 검토 | 채널별 수요예측 목적 |
| Stockout Prediction | SKU × 채널 × 센터 × 주차 | 센터 Grain 검토 필요 | 실제 결품사건은 센터 단위 |
| Reorder Decision | SKU × 채널 × 센터 × 주차 | SKU × 센터 × 주차 검토 | 채널 공용재고와 최종 발주 Grain |

### Grain Validation Finding 01 — Center inventory repetition

- 검증 대상:
  - SKU0001 × FC_DONGTAN × 2025-12-29

- 확인 결과:
  - GLOBAL_MALL 주문량: 5
  - ONLINE_MALL 주문량: 8
  - ROCKET_DELIVERY 주문량: 13
  - 각 채널 행의 available_qty: 138
  - 각 채널 행의 open_po_qty: 165

- 해석:
  - 판매수요는 SKU × 채널 × 센터 × 주차 Grain이다.
  - WMS 재고와 ERP PO는 SKU × 센터 × 주차 Grain이다.
  - 센터 공용재고와 PO가 채널별 Mart 행에 반복 표시된다.
  - 이는 JOIN Row Explosion은 아니지만, 채널 방향으로 SUM하면 재고와 PO가 과대집계된다.

- 현재 판정:
  - 분석용 Mart에서 참고값으로 반복하는 것은 가능하다.
  - 최종 발주추천에서 각 채널이 센터 전체 재고를 독립적으로 사용하는 것은 부적절할 수 있다.
  - Reorder Decision Grain을 SKU × 센터 × 주차로 분리할지 검토가 필요하다.

- 상태:
  - WARNING

  ### Reconciliation Finding 01 — OMS Raw to Weekly Mart

- Raw OMS order lines: 18,129
- Mapped OMS order lines: 18,129
- Mart order lines: 18,129

- Raw ordered quantity: 178,719
- Mart ordered quantity: 178,719
- Difference: 0

- Raw fulfilled quantity: 160,217
- Mart fulfilled quantity: 160,217
- Difference: 0

- Raw cancelled quantity: 7,878
- Mart cancelled quantity: 7,878
- Difference: 0

- 판정:
  - OMS Mapping 누락 없음
  - Mapping 중복으로 인한 Row Explosion 없음
  - 주간 GROUP BY 과정의 주문·출고·취소수량 손실 없음
  - Raw-to-Mart OMS Reconciliation PASS

- 한계:
  - 실제 외부 OMS와 Raw Table 간 Source-to-Raw 완전성은 검증 대상이 아님
  - 주문·취소·출고수량의 비즈니스 정의 적정성은 별도 Target 검증에서 확인

  ### Reconciliation Finding 02 — WMS Raw to Weekly Mart

- Raw snapshot rows: 6,240
- Mapped snapshot rows: 6,240
- Raw weekly keys: 6,240
- Mart center-week keys: 6,240

- Unexpected channel row count: 0
- Channel repeated value mismatch: 0
- Raw keys missing in Mart: 0
- Row-level value mismatches: 0

- Raw on-hand quantity: 920,165
- Mart on-hand quantity: 920,165
- Difference: 0

- Raw available quantity: 867,658
- Mart available quantity: 867,658
- Difference: 0

- Raw stockout events: 409
- Mart stockout events at center-week grain: 409
- Difference: 0

- 판정:
  - WMS Mapping 누락 및 중복 없음
  - SKU × 센터 × 주차 Grain 보존
  - Raw-to-Mart 재고수량 및 결품 Flag 일치
  - WMS Reconciliation PASS

- 주의:
  - 센터 재고와 결품 Flag는 채널별 Mart 행에 반복되므로 채널 방향 SUM 금지
  - Synthetic Data는 주차당 Snapshot 1개이므로 다중 Snapshot 최신값 선택 검증은 별도 필요

  ### Reconciliation Finding 04 — Promotion Raw to Weekly Mart

- Raw promotion events: 56
- Eligible promotion week candidates: 66
- Expected selected promotion keys: 66
- Mart selected promotion keys: 66
- Overlapping promotion keys: 0
- Flag mismatches: 0
- Value mismatches: 0

- 판정:
  - 프로모션 기간의 주차 전개 정상
  - SKU·채널 Mapping 정상
  - Mart 전달값 일치
  - Promotion Reconciliation PASS

- 주의:
  - 현재 데이터에는 동일 SKU × 채널 × 주차에 복수 프로모션이 존재하지 않아
    우선순위 선택 로직은 실제로 검증되지 않음
  - 현재 로직은 실제 영향력이 가장 큰 프로모션을 추정하는 것이 아니라
    업무 우선순위와 할인율에 따라 대표 프로모션 하나를 선택함
  - 운영형 모델에서는 promo_count, 유형별 flag, 실질 할인율,
    프로모션 중첩 여부를 추가하는 방식이 더 적절함

    ### Design Change Candidate 01 — Multi-Promotion Features

- 현재 구현:
  - SKU × 채널 × 주차별 대표 프로모션 1개만 선택
  - 선택 기준은 업무 우선순위, 할인율, promotion_id 순서

- 판정:
  - 데이터 정합성 검증은 PASS
  - 운영형 리테일 수요예측 설계로는 CHANGE REQUIRED

- 변경 방향:
  - Mart Grain은 유지한다.
  - 복수 프로모션을 여러 행으로 직접 JOIN하지 않는다.
  - SKU × 채널 × 주차 단위로 먼저 집계한 뒤 Feature로 결합한다.
  - primary_promo_type은 설명용으로 유지한다.
  - promo_count, promo_stack_flag, 유형별 flag,
    할인율 관련 feature를 추가한다.

- 추가 데이터 요구사항:
  - 프로모션 적용 가능 여부
  - 중복 적용 규칙
  - 프로모션 확정·승인시각
  - 할인 부담 주체
  - 쿠폰 최대 할인액 및 적용조건

- 테스트 데이터 요구사항:
  - 동일 SKU × 채널 × 주차에 2개 이상 프로모션이 겹치는 사례
  - 우선순위가 다른 중첩 사례
  - 동일 우선순위에서 할인율이 다른 사례
  - 중복 적용 가능·불가능 사례

- 상태:
  - CHANGE REQUIRED
  - 전체 현행 검증 완료 후 구현

  ### Audit Finding 05 — Demand Definition

- 현재 구현:
  - observed_sales_qty = fulfilled_qty
  - demand_gap_qty = ordered_qty - fulfilled_qty
  - stockout_adjusted_sales = fulfilled_qty + demand_gap_qty
  - 결과적으로 stockout_adjusted_sales = ordered_qty

- 검증 결과:
  - Current formula mismatch rows: 0
  - Adjusted vs ordered mismatch rows: 0
  - Gross order demand: 178,719
  - Net order demand: 170,841
  - Cancelled quantity: 7,878
  - Gross vs net overstatement: 4.61%
  - Current target vs net target different rows: 3,238

- 판정:
  - 계산식 구현은 정상
  - stockout_adjusted_sales라는 명칭과 실제 의미가 불일치
  - 현재 Target은 취소수량까지 포함한 Gross 주문량
  - 취소 사유 없이 Gross 주문량을 운영수요로 사용하는 것은 부적절할 수 있음

- 목표 설계:
  - 고객변심·결제실패 등 비수요 취소는 제외
  - 공급부족·재고부족 취소는 잠재수요에 포함
  - 취소사유 데이터가 없을 경우 Net 주문수요를 기본 후보로 비교

- 상태:
  - CHANGE REQUIRED
  - 운영 적용 전 Demand Target 재정의 필수

  ### Audit Finding 05 — Demand Definition

- Generator 확인 결과:
  - OMS cancelled_qty는 재고 부족과 무관하게 무작위로 생성된다.
  - 전체 취소는 2.5% 확률로 발생한다.
  - 나머지 주문은 0~4% 범위의 부분취소가 발생한다.
  - 취소 후 net_order_qty를 계산한 다음 가용재고와 비교해 fulfilled_qty가 결정된다.

- 실제 Synthetic 수요 정의:
  - net_order_demand = ordered_qty - cancelled_qty
  - unfulfilled_net_demand =
    max(net_order_demand - fulfilled_qty, 0)
  - stockout_adjusted_demand =
    fulfilled_qty + unfulfilled_net_demand
  - 결과적으로 stockout_adjusted_demand = net_order_demand

- 현재 구현 문제:
  - demand_gap_qty = ordered_qty - fulfilled_qty
  - 취소량과 재고 부족 미충족량을 구분하지 않는다.
  - stockout_adjusted_sales는 모든 행에서 ordered_qty와 같다.
  - 완전취소 주문도 실제 수요로 학습된다.

- 영향:
  - Gross 대비 Net 수요 과대계상: 7,878
  - Net 수요 대비 과대비율: 4.61%
  - Target이 달라지는 행: 3,238

- 판정:
  - 계산식 구현 일관성: PASS
  - Demand 업무 정의: FAIL
  - 운영형 수정 필요: CHANGE REQUIRED
  - 영향도: HIGH

  ### Audit Finding 06 — Target Alignment and Grain

- 시간축 검증:
  - SKU × 채널 × 센터 그룹 수: 360
  - 그룹별 주차 수: 52
  - 7일이 아닌 주차 간격: 0
  - Demand Target 불일치: 0
  - Stockout Target 불일치: 0
  - Boundary Null 처리 오류: 0

- Demand Target:
  - shift(-1) 시간 정렬은 정상
  - 현재 Target은 다음 주 Gross 주문량
  - 목표 Net 주문수요와 다른 행: 3,238
  - 업무 정의 변경 필요

- Stockout Target:
  - 채널 Grain Positive 행: 1,686
  - 센터 Grain Positive Key: 562
  - 반복비율: 3.0
  - 센터 결품사건이 채널 3개 행에 반복됨

- 판정:
  - Target 시간축 구현: PASS
  - Demand Target 정의: CHANGE REQUIRED / HIGH
  - Stockout Target Grain: CHANGE REQUIRED / HIGH

- 목표 방향:
  - Demand Forecast: SKU × 채널 × 센터 × 주차
  - Stockout Risk: SKU × 센터 × 주차
  - Reorder Decision: SKU × 센터 × 주차

  ### Audit Finding 07 — As-of and Feature Availability

- Decision cutoff:
  - Monday 08:00

- Validation result:
  - OMS orders after cutoff: 18,129 / 18,129
  - OMS after-cutoff ratio: 100%
  - Inventory snapshots exactly at cutoff: 6,240 / 6,240

- Critical issue:
  - Current-week realized sales features are used despite being unavailable
    at Monday 08:00.
  - The modeling row combines Monday inventory status with later weekly sales.

- Leakage features:
  - observed_sales_qty
  - demand_gap_qty
  - stockout_adjusted_sales
  - partial-fulfillment component of sales_censored_flag

- To-Be:
  - Use only completed prior-week sales history.
  - Split current_stockout_flag from lagged partial-fulfillment features.
  - Add confirmed/approved timestamps for promotion and override.
  - Add full timestamps for PO creation and confirmation.

- Status:
  - As-of Contract: FAIL
  - Severity: CRITICAL
  - Remediation: REQUIRED

  ### Audit Finding 08 — Lag and Rolling Features

- Calculation:
  - 모든 Lag·Rolling Feature의 독립 재계산 불일치: 0
  - shift(1) 이후 Rolling 계산 확인
  - 현재 주 실적이 자신의 과거 Feature에 포함되지 않음
  - 시간축 계산 PASS

- Null boundaries:
  - first-week rows: 360
  - lag_1w null rows: 360
  - lag_4w null rows: 1,440

- Definition findings:
  - sales rolling features는 실제 출고수량 기반
  - 결품 주에는 실제 수요를 과소표현할 수 있음
  - stockout_adjusted_sales rolling은 Gross 주문량 정의 오류를 승계
  - stockout_days_last_4w는 실제 일수가 아닌 결품 주차 수

- Early history:
  - 과거 1~3주 행: 1,080
  - 고정 분모와 동적 분모 차이: 0
  - 초기 결품 사례가 없어 분모 정책은 NOT TESTED

- Status:
  - Implementation: PASS
  - Adjusted demand rolling definition: CHANGE REQUIRED
  - Stockout feature naming: CHANGE REQUIRED
  - Early-history denominator: NOT TESTED

  ### Audit Finding 09 — Inventory, Cover and Safety Stock Grain

- Implementation:
  - Inventory position mismatch rows: 0
  - Inventory cover mismatch rows: 0
  - Safety stock mismatch rows: 0
  - Supply quantity invariant violations: 0

- Repetition:
  - Available quantity repetition ratio: 3.0
  - Inventory position repetition ratio: 3.0
  - Center inventory and PO values are repeated across three channels.

- Cover:
  - Center-week keys: 6,240
  - Keys with different channel covers: 5,936
  - Median center cover: 7.68 weeks
  - Median channel minimum cover: 15.76 weeks
  - Median channel maximum cover: 51.24 weeks

- Safety stock:
  - Keys with different channel safety stocks: 5,953
  - Current safety stock is calculated at channel grain despite shared center inventory.
  - Service-level coefficient is not defined.

- Additional concern:
  - All inbound within four weeks is included in inventory position regardless
    of the prediction and replenishment horizon.

- Status:
  - Formula implementation: PASS
  - Inventory cover grain: FAIL / CRITICAL
  - Safety stock grain: FAIL / CRITICAL
  - Horizon alignment: CHANGE REQUIRED

  ### Audit Finding 09 — Inventory, Cover and Safety Stock Grain

- Implementation:
  - Inventory position mismatch rows: 0
  - Inventory cover mismatch rows: 0
  - Safety stock mismatch rows: 0
  - Supply quantity invariant violations: 0

- Repetition:
  - Available quantity repetition ratio: 3.0
  - Inventory position repetition ratio: 3.0
  - Center inventory and PO values are repeated across three channels.

- Cover:
  - Center-week keys: 6,240
  - Keys with different channel covers: 5,936
  - Median center cover: 7.68 weeks
  - Median channel minimum cover: 15.76 weeks
  - Median channel maximum cover: 51.24 weeks

- Safety stock:
  - Keys with different channel safety stocks: 5,953
  - Current safety stock is calculated at channel grain despite shared center inventory.
  - Service-level coefficient is not defined.

- Additional concern:
  - All inbound within four weeks is included in inventory position regardless
    of the prediction and replenishment horizon.

- Status:
  - Formula implementation: PASS
  - Inventory cover grain: FAIL / CRITICAL
  - Safety stock grain: FAIL / CRITICAL
  - Horizon alignment: CHANGE REQUIRED

  ### Audit Finding 10 — Train/Test Label Horizon Overlap

- Demand:
  - Test start: 2025-11-03
  - Train rows before purge: 15,480
  - Overlap train rows: 360
  - Overlap feature weeks: 1
  - Train rows after purge: 15,120

- Stockout:
  - Test start: 2025-10-27
  - Train rows before purge: 15,120
  - Overlap train rows: 720
  - Overlap feature weeks: 2
  - Train rows after purge: 14,400

- Finding:
  - Current split uses feature week only.
  - It does not account for the future label horizon.
  - Training labels overlap the test-period boundary.

- Required design:
  - Demand model: purge 1 feature week before test.
  - Stockout model: purge 2 feature weeks before test.
  - Split function must explicitly receive target horizon.

- Status:
  - Chronological split implementation: PASS
  - Holdout temporal isolation: FAIL
  - Purged split: REQUIRED
  - Severity: HIGH

  ### Audit Finding 11 — Stockout Threshold and Baseline Fairness

- Threshold implementation:
  - DEFAULT_THRESHOLD = 0.5
  - FINAL_THRESHOLD = 0.4
  - Both thresholds are evaluated on the same test set.
  - Final prediction labels use threshold 0.4.

- Threshold governance:
  - No separate validation period is implemented.
  - No threshold search or business-cost function is implemented.
  - No evidence establishes that 0.4 was fixed before test evaluation.
  - Test tuning cannot be proven, but threshold independence is not demonstrated.

- Baseline:
  - Risk if inventory_cover_weeks < 2
    OR stockout_days_last_4w > 0.
  - Rule implementation is deterministic and correct.
  - Inventory cover uses the incorrect channel-level grain.
  - The two-week cover threshold has no documented operational basis.
  - stockout_days_last_4w represents weekly stockout snapshots, not days.

- Metric fairness:
  - Model ROC-AUC and PR-AUC use continuous probabilities.
  - Baseline ROC-AUC and PR-AUC use binary labels as scores.
  - Ranking metrics are not directly comparable.

- Required design:
  - Add Train / Validation / Test time splits with horizon purge.
  - Select threshold only on Validation.
  - Define threshold using false-negative cost, recall constraint,
    or operational alert capacity.
  - Lock threshold before Test evaluation.
  - Rebuild baseline at SKU × Center × Week grain.

- Status:
  - Threshold implementation: PASS
  - Threshold selection governance: FAIL / UNVERIFIED
  - Baseline fairness: FAIL
  - Severity: HIGH

  ### Audit Finding 12 — Demand Baseline Fairness

- Baseline:
  - sales_rolling_mean_4w
  - Missing values fallback to the training-target median
  - Predictions are clipped at zero

- Implementation:
  - Model and baseline use the same test rows.
  - Model and baseline use the same target and evaluation metrics.
  - Baseline uses prior completed-week sales history.
  - Baseline calculation is point-in-time safe.

- Fairness issues:
  - The model uses current-week realized features that are unavailable
    at the Monday 08:00 decision cutoff.
  - The baseline uses only completed historical information.
  - The target represents Gross ordered demand.
  - The baseline is based on fulfilled observed sales.
  - Therefore, model improvement over baseline cannot be interpreted fairly.

- Metric note:
  - forecast_accuracy is calculated as 1 - WAPE.
  - It should be labeled explicitly as WAPE-based accuracy.

- Required redesign:
  - Define Net Demand consistently.
  - Rebuild lag and rolling features from Net Demand.
  - Remove current-week realized features.
  - Apply horizon-purged Train / Validation / Test splits.
  - Compare against Last-Week, 4-Week Mean, and promo-aware baselines.

- Status:
  - Baseline implementation: PASS
  - Baseline time safety: PASS
  - Model-vs-baseline fairness: FAIL
  - Current model uplift claim: NOT VALID

  ### Audit Finding 13 — Demand Baseline Segment Performance

- Test rows: 2,880
- Fallback rows: 0
- Baseline prediction sum: 23,183.25
- Target sum: 25,806
- Aggregate bias: -2,622.75
- Bias rate: approximately -10.16%
- Overall WAPE: approximately 36.28%

- Segment findings:
  - No promotion / no stockout:
    - Rows: 2,754
    - WAPE: 35.42%
    - Bias: -2,186.75
  - No promotion / stockout:
    - Rows: 96
    - WAPE: 60.75%
    - Bias: -408.50
    - Baseline materially underforecasts constrained-demand periods.
  - Promotion / no stockout:
    - Rows: 30
    - WAPE: 41.16%
    - Sample size is too small for a strong conclusion.

- Notes:
  - No fallback value was used in the test period.
  - Current segmentation uses the feature-week promotion flag,
    while the target represents next-week demand.
  - Promotion effectiveness is therefore not directly measured.

- Status:
  - Baseline execution: PASS
  - Stockout-period robustness: FAIL
  - Promotion conclusion: INCONCLUSIVE

  ### Audit Finding 14 — Demand Feature Ablation

- Baseline:
  - WAPE: 36.2813%
  - Bias: -10.1633%

- Full original split:
  - WAPE: 32.9235%
  - Bias: -0.2704%

- Full model with horizon purge:
  - WAPE: 33.0243%
  - Bias: -1.5397%

- Current-week realized features removed with purge:
  - WAPE: 32.9439%
  - Bias: -1.6780%

- Findings:
  - Applying a one-week horizon purge increased WAPE by only 0.1008 percentage points.
  - Removing the four unavailable current-week realized features
    improved WAPE by 0.0804 percentage points.
  - These features violate the Monday 08:00 availability contract,
    but they did not materially inflate demand-model performance
    in the current synthetic dataset.
  - The reduced-feature model still improves WAPE over the
    four-week mean baseline by 3.3374 percentage points.

- Interpretation:
  - Temporal availability failure remains valid.
  - Performance inflation from the four removed features is not supported.
  - Other historical, categorical, promotion, inventory, or supply
    features appear to provide the remaining model uplift.

- Status:
  - Current-week feature availability: FAIL
  - Measured performance inflation from removed features: NOT OBSERVED
  - Safe-model uplift versus baseline: OBSERVED
  - Final production validity: NOT YET ESTABLISHED

  ### Audit Finding 15 — Demand Feature Group Dependence

- Base safe-model WAPE:
  - 32.9439%

- Group permutation results:
  - Demand history:
    - Permuted WAPE: 63.3812%
    - Increase: 30.4373 percentage points
  - Identity structure:
    - Permuted WAPE: 37.9766%
    - Increase: 5.0327 percentage points
  - Inventory and supply:
    - Permuted WAPE: 34.4387%
    - Increase: 1.4948 percentage points
  - Product static:
    - Increase: 0.8289 percentage points
  - Vendor and lead time:
    - Increase: 0.4311 percentage points
  - Promotion:
    - Increase: 0.0782 percentage points
  - Manual override:
    - No measured effect

- Interpretation:
  - The model depends primarily on historical demand features.
  - SKU, channel and center identities provide substantial additional signal.
  - Inventory features provide moderate signal, but their current
    contribution is contaminated by known grain errors.
  - Promotion importance is inconclusive because of sparse observations
    and horizon misalignment.
  - Manual override should likely be handled as a post-model policy rule.

- Cautions:
  - Permutation importance measures dependency, not causal or incremental uplift.
  - Group effects are not additive.
  - Demand history includes incorrectly defined adjusted-demand and
    stockout-history features.
  - Identity importance reflects known-entity forecasting and does not
    establish cold-start performance.

- Status:
  - Demand-history dependence: VERY HIGH
  - Identity dependence: HIGH
  - Inventory contribution: OBSERVED BUT UNRELIABLE
  - Promotion contribution: INCONCLUSIVE

  ### Audit Finding 16 — Demand Retrain Ablation

- Best-performing variant:
  - Clean demand history plus SKU, channel and center identity
  - WAPE: 32.8613%
  - Improvement versus four-week mean baseline: 3.4200 percentage points

- Full safe model:
  - 53 total features
  - WAPE: 32.9439%

- Clean-history-plus-identity model:
  - 9 total features
  - WAPE: 32.8613%
  - Simpler model slightly outperformed the full model.

- Identity removal:
  - WAPE increased from 32.9439% to 34.0486%
  - SKU, channel and center identities provide substantial predictive value.

- Inventory removal:
  - WAPE improved from 32.9439% to 32.8652%
  - Current inventory and supply features do not improve demand forecasting.
  - These features should be moved to the replenishment decision layer.

- Suspect-history removal:
  - WAPE improved from 32.9439% to 32.8798%
  - Incorrectly defined adjusted-demand and stockout-history features
    should be removed or rebuilt.

- Clean history only:
  - WAPE: 35.2565%
  - Historical demand features alone outperform the baseline,
    but most of the remaining uplift comes from entity identity.

- Status:
  - Feature simplification: REQUIRED
  - Inventory features in demand model: REMOVE
  - Suspect history features: REMOVE OR REDEFINE
  - Cold-start generalization: NOT TESTED

  ### Audit Finding 17 — Demand Cold-start Generalization

- Holdout:
  - Eight SKUs were excluded from the unseen-SKU training data.
  - Test rows: 576

- Known SKU with history and identity:
  - WAPE: 29.56%
  - Bias: -2.80%

- Unseen SKU with its own demand history:
  - WAPE: 31.47%
  - Bias: -5.88%
  - WAPE deterioration versus known SKU: 1.91 percentage points
  - Still outperformed the four-week history baseline.

- True cold-start using product and channel context only:
  - WAPE: 59.94%
  - Bias: -11.29%
  - WAPE deterioration versus known SKU: 30.37 percentage points

- Findings:
  - SKU identity provides meaningful predictive value.
  - Historical demand patterns still generalize to unseen SKU identities.
  - Existing performance is not explained solely by SKU memorization.
  - Current product-static and contextual features are insufficient
    for products with no demand history.
  - The history baseline is not a true cold-start baseline because
    it uses each holdout SKU's prior sales history.

- Required design:
  - Separate existing-SKU and new-product forecasting policies.
  - Build analogous-product and hierarchical cold-start baselines.
  - Add launch-plan, marketing, pricing and initial-allocation features.
  - Repeat the holdout experiment across multiple SKU samples.

- Status:
  - Existing-SKU forecasting: PROMISING
  - Unseen SKU with history: PARTIALLY GENERALIZABLE
  - True cold-start: FAIL
  - Robustness across SKU samples: NOT YET TESTED

  ### Design Decision — Lifecycle-based Forecast Routing

- Status:
  - PROPOSED
  - 반복 SKU 보류시험과 신규상품 데이터 설계 후 최종 확정

- Decision:
  - 예측 파이프라인은 하나로 운영한다.
  - 파이프라인 내부에서 SKU의 판매이력 충분성을 자동 판별한다.
  - SKU 상태에 따라 서로 다른 예측 경로를 적용한다.

- Proposed routes:
  - COLD_START: 판매이력이 부족한 완전 신규상품
  - WARM_UP: 일부 이력만 존재하는 초기상품
  - MATURE: 충분한 판매이력이 존재하는 기존상품

- Routing basis:
  - sales_history_weeks
  - nonzero_sales_weeks
  - valid_lag_count
  - first_sales_date
  - launch_date
  - active_flag

- Important rule:
  - new_product_flag 하나만으로 분기하지 않는다.
  - 4주·12주 기준은 임시 기준이며 데이터 검증 후 확정한다.

- Output contract:
  - 내부 경로가 달라도 최종 예측 결과 형식은 통일한다.
  - sku_lifecycle_stage
  - forecast_model_route
  - forecast_qty
  - forecast_confidence
  - fallback_used
  - reason_code

- Interpretation:
  - 별도 파이프라인 두 개를 독립 운영하는 구조가 아니다.
  - 하나의 통합 파이프라인 안에서 전문 예측경로를 자동 선택한다.

  ### Design Decision — Lifecycle-based Forecast Routing

- Status:
  - CONFIRMED IN PRINCIPLE

- Evidence:
  - 기존 SKU 모델 평균 WAPE: 32.38%
  - 미학습 SKU + 자체이력 평균 WAPE: 34.52%
  - 완전 신규 SKU 평균 WAPE: 58.53%
  - 완전 신규 SKU는 10회의 반복시험에서 지속적으로 불안정했다.

- Confirmed decision:
  - 하나의 통합 예측 파이프라인을 운영한다.
  - SKU 판매이력 충분성을 기준으로 예측경로를 자동 선택한다.
  - 기존상품과 완전 신규상품에 동일한 예측방식을 강제하지 않는다.

- Not yet fixed:
  - 신규·초기·기존 판정기준
  - 초기상품 예측 혼합비율
  - 신규상품 유사상품 선정기준
  - 신규상품 전용 Feature와 기준모델

  ### Audit Finding 18 — Stockout Output Grain Conflict

- Channel-grain test rows:
  - 2,880

- Unique SKU-center-week stockout events:
  - 960

- Row-to-event ratio:
  - 3.0

- Findings:
  - Every center-level stockout event is repeated across three channels.
  - All 960 center events received different channel-level probabilities.
  - Mean probability range across channels: 2.00 percentage points.
  - Maximum probability range: 98.25 percentage points.
  - At threshold 0.4, 21 center events received conflicting channel decisions.
  - At threshold 0.5, 20 center events received conflicting channel decisions.
  - The baseline rule also produced four channel-level disagreements.

- Event-level diagnostic:
  - Mean-probability threshold 0.4:
    - Precision: 78.26%
    - Recall: 78.26%
    - F1: 78.26%
  - Maximum-probability aggregation raised recall but increased false alerts.

- Required design:
  - Aggregate channel demand to SKU-center-week before stockout modeling.
  - Join center-level inventory and supply data at the same grain.
  - Train and output one stockout probability per SKU-center-week.
  - Select the probability threshold using a separate validation period.

- Status:
  - Current stockout modeling grain: FAIL
  - Current channel-level output: NOT OPERATIONALLY CONSISTENT
  - Center-level redesign: REQUIRED

  ### Audit Finding 19 — Stockout Counterfactual Conflict Drivers

- Original conflicting center events:
  - 21 of 960 events

- Current-week realized features equalized:
  - 14 of 21 original conflicts resolved
  - 7 conflicts remained
  - No new conflicts were created
  - Mean probability range on original conflicts decreased
    from 50.74 to 14.58 percentage points

- Inventory-derived features equalized:
  - No original conflicts were resolved
  - Association observed in the prior audit did not translate
    into direct conflict resolution

- Demand-history features equalized:
  - No original conflicts were resolved

- Channel identity fixed alone:
  - Four original conflicts were resolved
  - Ten new conflicts were created
  - Changing channel identity alone produced inconsistent
    feature combinations and is not a valid operational solution

- All numeric channel variation equalized:
  - Thirteen original conflicts were resolved
  - Eight remained
  - Channel identity and interaction effects remained

- All channel variation removed:
  - All 21 conflicts were resolved
  - Probability ranges became zero

- Interpretation:
  - Channel-grain input variation is the structural source of conflict.
  - Current-week realized features are the strongest measured direct driver.
  - Correlation with inventory-derived features did not establish causation.
  - The correct solution is center-grain model reconstruction,
    not post-model feature equalization.

- Status:
  - Current-week realized features: REMOVE
  - Channel-grain stockout model: REDESIGN
  - Center-grain reconstruction: REQUIRED

  ### Audit Finding 20 — Stockout Center-grain Reconstruction

- Source channel-grain rows:
  - 18,720

- Reconstructed SKU-center-week rows:
  - 6,240

- Row-to-center ratio:
  - 3.0

- Validation:
  - Duplicate center keys: 0
  - Events with fewer or more than three channels: 0
  - Target disagreement events: 0
  - Center-repeated field disagreements: 0

- Labeled center events:
  - 6,000

- Positive center-level stockout targets:
  - 562

- Historical feature reconstruction:
  - One-week lag null rows: 120
  - Four-week lag null rows: 480
  - Rolling-mean null rows: 120
  - Counts are consistent with 120 SKU-center time-series groups.

- Current-week realized features:
  - Four unavailable features were excluded from the model candidate set.
  - Historical sales features were rebuilt after aggregating channel sales
    to the center level.

- Remaining cautions:
  - Historical sales currently use fulfilled sales rather than corrected
    net demand.
  - Promotion features remain provisional due to forecast-horizon and
    confirmation-time issues.

- Status:
  - Center-grain reconstruction: PASS
  - Center-level field consistency: PASS
  - Model training readiness: CONDITIONAL PASS

  ### Audit Finding 21 — Center-grain Stockout Model

- Center-grain model:
  - Validation-selected threshold: 0.20
  - Validation F1: 65.31%
  - Test F1: 50.53%
  - Test precision: 48.98%
  - Test recall: 52.17%
  - Test PR-AUC: 54.94%
  - Test ROC-AUC: 96.77%

- Threshold transfer:
  - Validation-to-test F1 decreased by 14.78 percentage points.
  - The probability ranking remained strong, but classification
    performance was not stable across periods.

- Baseline comparison:
  - Model false positives: 25
  - Model false negatives: 22
  - Baseline false positives: 80
  - Baseline false negatives: 13
  - The preferred method depends on the relative cost of missed
    stockouts and unnecessary alerts.

- Split defect:
  - The code specifies an eight-week validation window.
  - The effective validation set contains only six weeks because
    two weeks are removed by the test-boundary label purge.
  - Threshold selection must be repeated using eight effective
    validation weeks.

- Status:
  - Center-grain reconstruction: PASS
  - Model ranking ability: PROMISING
  - Threshold stability: FAIL
  - Operational threshold: NOT ESTABLISHED

  ### Audit Finding 22 — Corrected Threshold Stability

- Corrected split:
  - Train: 30 weeks
  - Validation: 8 effective weeks
  - Test: 8 weeks
  - Two-week label purge was applied at both boundaries.

- Validation-selected threshold:
  - 0.20

- Validation performance:
  - Precision: 64.20%
  - Recall: 72.22%
  - F1: 67.97%

- Test performance:
  - Precision: 52.50%
  - Recall: 45.65%
  - F1: 48.84%
  - PR-AUC: 54.09%
  - ROC-AUC: 96.54%

- Threshold transfer:
  - F1 decreased by 19.14 percentage points.
  - The corrected eight-week validation window did not solve
    threshold instability.

- Temporal behavior:
  - Positive rates declined from 11.36% in training
    to 7.50% in validation and 4.79% in testing.
  - Weekly precision and recall changed materially despite
    consistently strong ROC-AUC.

- Cost comparison:
  - Model: 19 false positives and 25 false negatives.
  - Baseline: 80 false positives and 13 false negatives.
  - The break-even missed-stockout cost ratio is approximately 5.08.

- Status:
  - Split implementation: PASS
  - Ranking ability: PROMISING
  - Static threshold stability: FAIL
  - Operational model selection: COST-DEPENDENT

  ### Audit Finding 23 — Rolling Threshold Instability

- Selected thresholds across four rolling folds:
  - 0.45
  - 0.55
  - 0.25
  - 0.20

- Threshold stability:
  - Mean: 0.3625
  - Standard deviation: 0.1652
  - Range: 0.20 to 0.55

- Validation-to-test transfer:
  - Every fold showed a negative F1 transfer gap.
  - Mean validation F1: 66.24%
  - Mean test F1: 55.30%
  - Mean transfer gap: -10.95 percentage points

- Fixed threshold result:
  - Threshold 0.10 produced the highest mean test F1:
    56.07%
  - Its test F1 standard deviation was 4.43 percentage points.
  - It slightly outperformed the dynamically selected thresholds.

- Interpretation:
  - Validation-optimized thresholds are overfitting individual periods.
  - Ranking performance remains strong.
  - Probability and binary decision stability remain insufficient.
  - Thresholds from 0.05 to 0.15 form a comparatively stable region.

- Status:
  - Ranking ability: PROMISING
  - Validation-selected threshold policy: FAIL
  - Fixed threshold 0.10: TECHNICAL CANDIDATE ONLY
  - Operational threshold: COST-DEPENDENT
  - Ranking-based alert policy: RECOMMENDED FOR TESTING

  ### Audit Finding 24 — Rank-based Alert Policy

- Best balanced policy:
  - Global weekly top 10 alerts

- Performance:
  - Mean precision: 48.44%
  - Mean recall: 68.95%
  - Mean F1: 56.14%
  - F1 standard deviation: 4.22 percentage points
  - Alerts per week: exactly 10

- Comparison with fixed threshold 0.10:
  - Mean F1 was nearly identical.
  - Rank-based policy produced a fixed operational workload.
  - Fixed-threshold alerts varied by week and reached a maximum of 22.

- Capacity alternatives:
  - Top 5%: precision-oriented, six alerts per week
  - Top 10: balanced, ten alerts per week
  - Top 10%: recall-oriented, twelve alerts per week

- Center-level quota:
  - Center-based top-percent policies underperformed
    the corresponding global policies.
  - Forced center allocation may spend alert capacity
    on relatively low-risk events.

- Status:
  - Ranking-based alert policy: PASS
  - Global weekly top 10: PRIMARY TECHNICAL CANDIDATE
  - Fixed probability threshold: SECONDARY REFERENCE
  - Final alert capacity: BUSINESS DECISION REQUIRED

  

