# 02. Data Source Design

## 1. 문서 목적

본 문서는 `Retail InStock Decision MVP`의 원천 시스템 가정, 파일별 역할, 주요 필드, 데이터 통합 원칙 및 SQL Data Mart 구조를 정의한다. 원천 데이터는 하나의 분석용 CSV가 아니라 실제 리테일 환경처럼 여러 업무 시스템에 분산된 형태로 설계한다.

## 2. Data Architecture Overview

데이터 흐름은 아래 4개 Layer로 구분한다.

1. **Source Layer**: 시스템별 Synthetic CSV
2. **Staging Layer**: 원천 구조를 보존하여 SQLite에 적재한 테이블
3. **Mart Layer**: 코드, 시간, Grain을 정합화한 SQL Data Mart
4. **Serving Layer**: ML 학습·추론 및 Power BI용 Dataset

모든 데이터는 최종적으로 **SKU × 채널 × 센터 × 주차** Grain에 연결된다. 단, 원천 테이블 자체의 Grain은 업무 목적에 따라 다르며 Staging 단계에서 강제로 동일 Grain으로 변형하지 않는다.

## 3. Source System Inventory

| Source domain | File | 원천 Grain | 주요 역할 |
|---|---|---|---|
| Product Master | `sku_master.csv` | SKU | 상품 속성 및 발주 제약 |
| Vendor Master | `vendor_master.csv` | Vendor | 벤더 계약·공급 특성 |
| Code Mapping | `sku_code_mapping.csv` | System × source SKU code | 시스템별 상품 코드 표준화 |
| OMS | `oms_sales_orders.csv` | Sales order line | 주문·판매·취소 이력 |
| WMS | `wms_inventory_snapshot.csv` | SKU × Center × Snapshot time | 재고 상태 스냅샷 |
| WMS | `wms_goods_receipts.csv` | Receipt line | 실제 입고 이력 |
| ERP | `erp_purchase_orders.csv` | PO line | 발주 계획과 상태 |
| MD Calendar | `md_promotion_calendar.csv` | Promotion × SKU/Channel | 프로모션 계획 |
| Manual | `manual_overrides.csv` | 적용 대상 × 유효기간 | 담당자 예외 규칙 |

## 4. Source File Design

아래 필드는 MVP 구현을 위한 권장 최소 설계다. 모든 날짜·시간은 ISO 형식을 사용하고, 수량은 기본 판매 단위(Base Unit)를 기준으로 한다.

### 4.1 Product Master: `sku_master.csv`

**Purpose**  
표준 SKU와 상품 속성, 기본 벤더, 발주 제약을 관리한다.

**Source Grain**  
SKU당 1행. 유효기간 이력을 구현할 경우 SKU × effective period당 1행.

| Field | 설명 |
|---|---|
| `sku_id` | 내부 표준 SKU Key |
| `sku_name` | 상품명 |
| `category_l1`, `category_l2` | 상품 카테고리 |
| `brand` | 브랜드 |
| `default_vendor_id` | 기본 공급 벤더 |
| `unit_cost` | 단위 원가 |
| `list_price` | 정상 판매가 |
| `shelf_life_days` | 유통기한 또는 사용기한 |
| `launch_date` | 출시일 |
| `discontinue_date` | 단종일, 미정이면 Null |
| `moq_qty` | 최소발주수량(Minimum Order Quantity) |
| `order_multiple` | 발주 배수 |
| `min_order_amount` | 최소발주금액 |
| `active_flag` | 운영 대상 여부 |

### 4.2 Vendor Master: `vendor_master.csv`

**Purpose**  
벤더별 기준 리드타임과 공급 조건을 관리한다.

**Source Grain**  
Vendor당 1행. 계약 이력 반영 시 Vendor × effective period당 1행.

| Field | 설명 |
|---|---|
| `vendor_id` | 벤더 Key |
| `vendor_name` | 벤더명 |
| `standard_lead_time_days` | 계약 기준 리드타임 |
| `order_cycle_days` | 정기 발주 주기 |
| `payment_terms` | 결제 조건 |
| `supply_region` | 공급 지역 |
| `vendor_active_flag` | 거래 가능 여부 |

실제 리드타임 평균, 편차, 납기 준수율은 Master 값이 아니라 PO와 입고 실적에서 계산한다.

### 4.3 Code Mapping: `sku_code_mapping.csv`

**Purpose**  
OMS, WMS, ERP 등 시스템별 상이한 상품 코드를 표준 `sku_id`에 연결한다.

**Source Grain**  
Source system × source SKU code × effective period당 1행.

| Field | 설명 |
|---|---|
| `source_system` | `OMS`, `WMS`, `ERP`, `MD` 등 |
| `source_sku_code` | 원천 시스템 상품 코드 |
| `sku_id` | 내부 표준 SKU Key |
| `effective_from`, `effective_to` | 매핑 유효기간 |
| `mapping_status` | Valid, Pending, Invalid 등 |

매핑되지 않거나 중복 매핑된 코드는 조용히 제외하지 않고 Data Quality 예외로 분리한다.

### 4.4 OMS: `oms_sales_orders.csv`

**Purpose**  
주문, 출고 및 취소 이력을 이용해 관측 판매량을 계산한다.

**Source Grain**  
Sales order line당 1행.

| Field | 설명 |
|---|---|
| `order_id`, `order_line_id` | 주문 및 라인 식별자 |
| `order_datetime` | 주문 시각 |
| `channel_id` | 판매 채널 |
| `fulfillment_center_id` | 출고 예정 또는 실제 센터 |
| `oms_sku_code` | OMS 상품 코드 |
| `ordered_qty` | 주문수량 |
| `fulfilled_qty` | 출고 완료수량 |
| `cancelled_qty` | 취소수량 |
| `unit_selling_price` | 실판매 단가 |
| `order_status` | 주문 상태 |

기본 관측 판매량은 비즈니스 정의에 따라 `fulfilled_qty` 또는 취소를 제외한 유효 주문수량으로 집계하며, 한 가지 기준을 Mart 전체에 일관되게 적용한다.

### 4.5 WMS: `wms_inventory_snapshot.csv`

**Purpose**  
센터별 On-hand, 예약, 가용, 품질보류 재고 및 품절 상태를 시점별로 기록한다.

**Source Grain**  
WMS SKU code × Center × Snapshot datetime당 1행.

| Field | 설명 |
|---|---|
| `snapshot_datetime` | 스냅샷 기준 시각 |
| `center_id` | 물류센터 Key |
| `wms_sku_code` | WMS 상품 코드 |
| `on_hand_qty` | 장부상 보유수량 |
| `reserved_qty` | 주문 등에 할당된 수량 |
| `damaged_qty` | 파손 또는 사용 불가 수량 |
| `quality_hold_qty` | 품질보류 수량 |
| `available_qty` | 즉시 판매 가능한 수량 |
| `stockout_flag` | 해당 시점 가용재고 소진 여부 |

`available_qty`는 원천 제공값을 우선 사용하되, 재계산 시에는 `on_hand_qty - reserved_qty - damaged_qty - quality_hold_qty`와 정합성을 검사한다.

### 4.6 WMS: `wms_goods_receipts.csv`

**Purpose**  
PO별 실제 입고 시점과 수량을 기록하여 리드타임 및 납품 이행률을 계산한다.

**Source Grain**  
Goods receipt line당 1행. 한 PO line의 분할 입고를 허용한다.

| Field | 설명 |
|---|---|
| `receipt_id`, `receipt_line_id` | 입고 식별자 |
| `po_id`, `po_line_id` | 연결된 발주 라인 |
| `receipt_datetime` | 실제 입고 처리 시각 |
| `center_id` | 입고 센터 |
| `wms_sku_code` | WMS 상품 코드 |
| `received_qty` | 실제 입고수량 |
| `accepted_qty` | 검수 합격수량 |
| `rejected_qty` | 검수 불합격수량 |
| `receipt_status` | 입고 상태 |

미래 실제 입고일과 수량은 과거 시점 Feature에 사용할 수 없다. 과거 Feature에는 그 시점까지 완료된 입고 실적만 사용한다.

### 4.7 ERP: `erp_purchase_orders.csv`

**Purpose**  
발주 생성, 약속 납기, 발주수량 및 잔량을 관리한다.

**Source Grain**  
PO line당 1행. 상태 변경 이력을 보존하려면 PO line × status snapshot당 1행.

| Field | 설명 |
|---|---|
| `po_id`, `po_line_id` | 발주 및 라인 식별자 |
| `po_created_date` | 발주 생성일 |
| `vendor_id` | 공급 벤더 |
| `erp_sku_code` | ERP 상품 코드 |
| `center_id` | 납품 대상 센터 |
| `ordered_qty` | 발주수량 |
| `confirmed_qty` | 벤더 확정수량 |
| `promised_delivery_date` | 당시 약속 납기일 |
| `po_status` | Open, Partial, Closed, Cancelled 등 |
| `cancelled_qty` | 취소수량 |
| `last_updated_at` | 원천 최종 변경 시각 |

향후 입고 Feature는 Decision Cutoff 당시 Open PO의 확정수량과 당시 약속 납기만 사용한다. 이후 수정된 납기일로 과거 상태를 재작성하면 Leakage가 발생하므로 Snapshot 또는 변경 이력 보존이 필요하다.

### 4.8 MD Calendar: `md_promotion_calendar.csv`

**Purpose**  
사전에 계획된 프로모션 일정, 유형 및 할인 조건을 관리한다.

**Source Grain**  
Promotion × source SKU code × Channel당 1행.

| Field | 설명 |
|---|---|
| `promotion_id` | 프로모션 Key |
| `md_sku_code` | MD 시스템 상품 코드 |
| `channel_id` | 대상 채널 |
| `promo_type` | 쿠폰, 가격할인, 기획전 등 |
| `promo_start_date`, `promo_end_date` | 계획된 행사 기간 |
| `planned_discount_rate` | 계획 할인율 |
| `planned_promo_price` | 계획 판매가 |
| `promo_priority` | 프로모션 중요도 또는 노출 등급 |
| `calendar_updated_at` | 계획 최종 변경 시각 |

Feature에는 Decision Cutoff 시점에 공지된 계획 정보와 과거 종료 프로모션의 실적만 사용한다. 진행 중인 프로모션의 종료 후 실제 Uplift는 사용할 수 없다.

### 4.9 Manual: `manual_overrides.csv`

**Purpose**  
모델이 알 수 없는 단종, 공급중단, 행사 대응 및 담당자 승인 예외를 Rule Engine에 반영한다.

**Source Grain**  
Override rule당 1행.

| Field | 설명 |
|---|---|
| `override_id` | Override Key |
| `sku_id` | 대상 SKU, 전체 적용 시 Null 허용 |
| `channel_id`, `center_id` | 대상 채널·센터, 전체 적용 시 Null 허용 |
| `effective_from`, `effective_to` | 적용 기간 |
| `override_type` | Force Hold, Block Buy, Expedite 등 |
| `override_value` | 수량, 임계값 또는 상태 값 |
| `override_reason` | 적용 사유 |
| `created_at`, `created_by` | 생성 시점 및 담당자 |
| `approval_status` | 승인 상태 |

Override는 모델 학습 Label을 변경하는 용도가 아니라 최종 의사결정 단계의 통제 규칙으로 적용하고, 적용 전·후 결과를 모두 추적한다.

## 5. Integration and Transformation Rules

### 5.1 Standard Key

- 시스템별 상품 코드는 `sku_code_mapping.csv`의 유효기간 조건으로 `sku_id`에 매핑한다.
- 채널과 센터 코드는 전 Source에서 동일한 표준 코드 체계를 사용하거나 별도 표준 매핑을 적용한다.
- PO와 입고는 `po_id + po_line_id`로 연결하여 분할 입고를 집계한다.

### 5.2 Time Standardization

- 모든 Timestamp는 원천 Timezone을 보존한 뒤 분석 표준 Timezone으로 변환한다.
- 주차 기준은 하나의 `week_start_date` 규칙으로 고정한다.
- Decision Cutoff 이후 생성되거나 수정된 레코드는 해당 과거 주차 Feature에서 제외한다.
- 날짜가 겹치는 Promotion과 Override는 명시된 우선순위 규칙으로 해소한다.

### 5.3 Quantity and Status Rules

- 판매, 재고, 입고, 발주 수량은 동일한 Base Unit으로 변환한다.
- 취소 PO, 취소 주문, 불합격 입고는 가용 공급 또는 유효 판매에서 제외한다.
- Open PO 잔량은 확정수량에서 Cutoff 시점까지의 누적 유효 입고와 취소수량을 차감하여 계산한다.
- Null, 0, 미수집 값을 구분하며 미수집 값을 0으로 자동 대체하지 않는다.

## 6. SQL Data Mart Design

### 6.1 Mart Grain and Key

최종 Feature Mart의 Primary Key는 다음 조합이다.

```text
sku_id + channel_id + center_id + week_start_date
```

Mart는 각 주차 Decision Cutoff 시점의 스냅샷을 표현한다. 학습용 과거 행과 운영 추론용 최신 행이 동일한 컬럼 정의를 공유해야 한다.

### 6.2 Recommended Mart Structure

| Mart / View | Grain | 역할 |
|---|---|---|
| `dim_sku` | SKU | 상품 및 발주 제약 |
| `dim_vendor` | Vendor | 벤더 기준 속성 |
| `dim_calendar` | Date / Week | 날짜 및 주차 기준 |
| `fct_sales_weekly` | SKU × Channel × Center × Week | 주간 판매 및 취소 집계 |
| `fct_inventory_weekly` | SKU × Center × Week | Cutoff 재고 및 주간 품절 집계 |
| `fct_po_weekly` | SKU × Center × Vendor × Week | 발주, 잔량 및 예정 입고 집계 |
| `fct_receipts_weekly` | SKU × Center × Vendor × Week | 입고 및 공급 실적 집계 |
| `fct_promotion_weekly` | SKU × Channel × Week | 프로모션 계획·과거 실적 집계 |
| `mart_retail_instock_weekly` | SKU × Channel × Center × Week | 모델 및 의사결정 통합 Mart |

센터 단위 재고·PO를 채널별로 사용할 때는 채널 전용 재고인지 공용 재고인지 구분해야 한다. 공용 재고라면 과거 출고 비중 등 사전에 정의한 배부 규칙과 배부 근거 컬럼을 Mart에 남긴다.

### 6.3 Target Build

- `target_demand_next_1w`: 동일 Grain의 다음 주 `stockout_adjusted_sales`를 Lead하여 생성
- `target_stockout_risk_next_2w`: 동일 Grain에서 다음 2개 주차 중 결품 위험 조건이 한 번이라도 참이면 `1`

Target 산출에 사용한 미래 데이터는 Label 컬럼에만 존재해야 하며 Feature Query와 물리적·논리적으로 분리한다. 운영 추론용 최신 행의 Target은 Null이다.

## 7. Data Quality Controls

| Check | 검증 내용 | 처리 원칙 |
|---|---|---|
| Key uniqueness | Mart Primary Key 중복 여부 | 중복 원인 해소 전 적재 실패 처리 |
| Code mapping | 미매핑·다중 매핑 SKU | 예외 테이블 격리 및 건수 보고 |
| Quantity validity | 음수 또는 비정상 대량 수량 | 반품 등 허용 사유가 없으면 오류 처리 |
| Inventory equation | 가용재고와 구성 수량 정합성 | 허용 오차 초과 시 품질 Flag |
| PO-receipt linkage | 입고의 PO 참조 유효성 | 미연결 입고 별도 관리 |
| Temporal validity | 생성·수정 시각이 Cutoff 이전인지 | 미래 정보 Feature 제외 |
| Promotion overlap | 중복 행사 및 할인율 충돌 | 우선순위 규칙 적용 및 Flag 유지 |
| Referential integrity | SKU, Vendor, Center 참조 무결성 | Orphan Record 격리 |

데이터 품질 Flag는 단순 삭제 기준이 아니라 모델 Fallback 또는 Power BI 경고에 활용할 수 있도록 보존한다.

## 8. Leakage Prevention in Data Layer

Data Layer에서는 Point-in-Time Correct Join을 원칙으로 한다.

- 미래 실제 판매량은 Target 생성 외에는 Join하지 않는다.
- 미래 실제 입고일·수량은 당시 Open PO의 예정 정보로 대체할 수 없으며 Feature에서 제외한다.
- 미래 실제 품절 여부는 결품 Target 생성에만 사용한다.
- 현재 또는 미래 프로모션의 종료 후 실제 Uplift를 과거 Feature로 역반영하지 않는다.
- PO 납기, 프로모션 일정, Override는 Decision Cutoff 당시 유효했던 Version을 사용한다.
- 집계 Window는 항상 현재 행의 Decision Cutoff에서 닫히도록 설정한다.

## 9. Serving Outputs

### 9.1 ML Dataset

- Decision Key와 기준일
- Feature 컬럼
- 학습 행에 한정된 Target 컬럼
- Data Quality 및 수집 가능성 Flag

### 9.2 Power BI Dataset

- Decision Key와 상품·벤더 설명 정보
- 현재 재고, 예정 입고, 예측 수요, 결품 위험 확률
- `recommended_order_qty`, `recommended_action`, `action_reason`
- 주요 판단 Feature와 Data Quality Flag
- 모델 Version, Rule Version, 생성 Timestamp

Power BI Dataset은 결과 조회용 Serving Layer이며, 대시보드에서 원천 로직을 다시 계산하지 않는 것을 원칙으로 한다.
