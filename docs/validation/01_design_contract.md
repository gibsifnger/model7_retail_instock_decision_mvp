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