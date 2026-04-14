# Local AI Handoff

작성일: 2026-04-14  
작성 위치: `/Users/a0067/Documents/Playground/이거넣어 코랩/LOCAL_AI_HANDOFF_20260414.md`

이 문서는 이 프로젝트를 GitHub에 올린 뒤, 다른 로컬 AI나 코딩 에이전트가 현재 상태를 빠르게 이해하고 이어서 작업할 수 있도록 만든 최신 인수인계 문서다.

이 문서를 먼저 읽고, 그 다음 아래 업로드 폴더의 최신 스크립트를 기준으로 판단하는 것을 권장한다.

- 코랩 업로드용 최신 폴더: [`/Users/a0067/Documents/Playground/이거넣어 코랩/진짜 이거 넣어 코랩`](/Users/a0067/Documents/Playground/이거넣어 코랩/진짜 이거 넣어 코랩)

## 1. 프로젝트 목적

목표는 서울시 `500m x 500m` 격자 단위 보행자-차량 사고 위험을 시공간적으로 예측하는 것이다.

현재 최종 방향은 다음과 같다.

- 딥러닝 학습 단위: 월단위
- 모델 구조: ConvLSTM
- 입력: 최근 `24개월`
- 직접 예측 타깃: `다음 1개월` 사고 발생 여부
- 미래 예측 방식: `1-step recursive forecasting`
- 최종 산출: `2025`, `2026`, `2027` 각 연도 위험도

즉, 과거 24개월을 보고 다음 1개월을 예측하는 모델을 학습한 뒤, 이를 2025-01부터 2027-12까지 재귀적으로 굴려 월별 확률을 만들고, 그 월별 확률을 연단위로 집계하는 구조다.

## 2. 왜 이 구조로 바꿨는가

이 프로젝트는 여러 단계를 거쳤다.

### 과거 시도

- 연단위 직접 예측
- 월단위 학습 후 다음 12개월 동시 예측
- 월학습 + 연집계
- threshold / calibration 후처리 위주 개선

### 현재 판단

이전 구조는 아래 문제가 컸다.

- PR-AUC가 낮음
- Precision이 낮음
- 확률 calibration이 불안정함
- `0.99~1.00` 같은 과도한 확률이 자주 나옴
- 12개월 동시 예측은 표본 수와 목표 일관성 측면에서 불리함

그래서 현재는 더 단순하고 학습 목표가 분명한 구조로 재구성했다.

- `24개월 입력 -> 다음 1개월 예측`
- 월별 예측을 연도별로 나중에 집계
- 최근성, hotspot 전이, 노출 대비 위험도를 feature로 직접 강화

## 3. 현재 기준 최신 핵심 파일

현재 코랩 업로드 폴더 기준으로 가장 중요한 파일은 아래 5개다.

- [build_convlstm_monthly_tensors_from_grid_panel.py](/Users/a0067/Documents/Playground/이거넣어 코랩/진짜 이거 넣어 코랩/build_convlstm_monthly_tensors_from_grid_panel.py)
- [train_convlstm_monthly_colab_gpu.py](/Users/a0067/Documents/Playground/이거넣어 코랩/진짜 이거 넣어 코랩/train_convlstm_monthly_colab_gpu.py)
- [forecast_monthly_yearly_one_step_colab.py](/Users/a0067/Documents/Playground/이거넣어 코랩/진짜 이거 넣어 코랩/forecast_monthly_yearly_one_step_colab.py)
- [evaluate_monthly_annual_one_step_colab.py](/Users/a0067/Documents/Playground/이거넣어 코랩/진짜 이거 넣어 코랩/evaluate_monthly_annual_one_step_colab.py)
- [feature_spec_extensions.json](/Users/a0067/Documents/Playground/이거넣어 코랩/진짜 이거 넣어 코랩/feature_spec_extensions.json)

### 각 파일 역할

`build_convlstm_monthly_tensors_from_grid_panel.py`
- 주간 패널 + 사고 event + 외부 feature를 합쳐 월 패널 생성
- row/col 매핑 생성
- lag / history / spatial / exposure 파생 feature 생성
- 24개월 입력, 1개월 타깃 텐서 생성
- train/valid/test split 저장

`train_convlstm_monthly_colab_gpu.py`
- 최신 1-step ConvLSTM 학습 스크립트
- 2-layer ConvLSTM + dropout head
- focal + OHEM 계열 loss
- weighted sampler
- AdamW
- gradient clipping
- validation threshold 탐색 기반 checkpoint 선택

`forecast_monthly_yearly_one_step_colab.py`
- temperature scaling 계산
- 2025-01 ~ 2027-12까지 월별 recursive forecasting
- `monthly_prediction_probabilities_2025_2027.csv` 생성
- `annual_prediction_raw_2025_2027.csv` 생성

`evaluate_monthly_annual_one_step_colab.py`
- validation/test 월별 확률 평가
- validation에서 threshold 탐색
- monthly calibration method 선택
- annual calibration method 선택
- 최종 `annual_prediction_calibrated_2025_2027.csv` 생성
- `eval_summary.json`과 그래프/CSV 산출

`feature_spec_extensions.json`
- 확장 변수 on/off
- 변수 메타데이터
- 원천 기간과 상태 기록

## 4. 코랩 업로드 최소 세트

현재 최신 파이프라인을 돌리기 위한 코랩 업로드 최소 세트는 아래 `11개 파일`이다.

- `build_convlstm_monthly_tensors_from_grid_panel.py`
- `train_convlstm_monthly_colab_gpu.py`
- `forecast_monthly_yearly_one_step_colab.py`
- `evaluate_monthly_annual_one_step_colab.py`
- `feature_spec_extensions.json`
- `grid_external_features.csv`
- `grid_month_dynamic_features_additional.csv`
- `grid_year_features_additional.csv`
- `seoul_grid_500m.gpkg`
- `seoul_grid_week_full_panel_enriched_min.csv`
- `seoul_pedestrian_vehicle_events_2007_2024_with_lonlat.csv`

주의:
- 실질적으로는 `11개`다. 과거 대화에서 9개 기준으로 말하던 시점이 있었지만, 현재 최신 1-step 파이프라인은 `forecast`와 `evaluate` 스크립트가 추가되어야 한다.
- 코랩 fresh runtime에서는 중간 산출물 폴더를 업로드하지 않는다.

## 5. 데이터 개요

- 공간 단위: 서울시 500m 격자
- 격자 수: `2,634`
- 설명변수 기준 기간: `2005-01 ~ 2024-12`
- 사고 event 원천 기간: `2007-01-01 ~ 2024-12-31`
- 미래 예측 기간: `2025-01 ~ 2027-12`

핵심 원천 파일:

- [seoul_grid_week_full_panel_enriched_min.csv](/Users/a0067/Documents/Playground/이거넣어 코랩/진짜 이거 넣어 코랩/seoul_grid_week_full_panel_enriched_min.csv)
- [seoul_pedestrian_vehicle_events_2007_2024_with_lonlat.csv](/Users/a0067/Documents/Playground/이거넣어 코랩/진짜 이거 넣어 코랩/seoul_pedestrian_vehicle_events_2007_2024_with_lonlat.csv)
- [grid_external_features.csv](/Users/a0067/Documents/Playground/이거넣어 코랩/진짜 이거 넣어 코랩/grid_external_features.csv)
- [grid_month_dynamic_features_additional.csv](/Users/a0067/Documents/Playground/이거넣어 코랩/진짜 이거 넣어 코랩/grid_month_dynamic_features_additional.csv)
- [grid_year_features_additional.csv](/Users/a0067/Documents/Playground/이거넣어 코랩/진짜 이거 넣어 코랩/grid_year_features_additional.csv)

## 6. 현재 모델 입력 특징량 구조

현재 최신 빌드 스크립트 기준 `load_feature_spec()` 결과:

- sequence length: `24`
- prediction horizon: `1`
- target col: `target_next_month`
- feature spec count: `64`

중요:
- 예전 문서의 `29개`, `33개`, `37개` 같은 숫자는 현재 최신 버전 기준으로 이미 outdated일 수 있다.
- 앞으로는 노트북에서 `feature_count == 33` 같은 고정 assert를 두면 안 된다.
- 반드시 빌드 산출물 `build_summary.json`과 `feature_spec_resolved.json`의 실제 값을 기준으로 확인해야 한다.

### feature 종류

현재 feature kind:

- `dynamic_monthly`
- `static_grid`
- `yearly_expand`
- `derived_lag`
- `derived_history`
- `derived_calendar`
- `derived_spatial`
- `derived_exposure`

### 최근 추가한 핵심 파생 feature

현재 실무성 개선을 위해 아래 feature들을 추가했다.

- 장기 lag:
  - `lag_6m`
  - `lag_12m`

- rolling history:
  - `recent_3m_mean`
  - `recent_6m_mean`
  - `recent_12m_mean`
  - `recent_24m_sum`
  - `recent_24m_mean`
  - `recent_6m_max`
  - `recent_12m_max`
  - `recent_6m_std`
  - `recent_12m_std`
  - `recent_3m_minus_prev_3m`
  - `recent_6m_minus_prev_6m`

- spatial hotspot:
  - `neighbor_lag_1m_mean`
  - `neighbor_lag_3m_mean`
  - `neighbor_lag_12m_mean`
  - `neighbor_recent_3m_mean`
  - `neighbor_recent_6m_mean`
  - `neighbor_recent_12m_mean`
  - `neighbor_recent_24m_mean`

- exposure-normalized risk:
  - `lag_1m_per_traffic`
  - `recent_3m_per_traffic`
  - `recent_12m_per_traffic`
  - `recent_3m_per_weekday_pop`
  - `recent_12m_per_weekday_pop`

### 현재 extension 변수 상태

현재 `feature_spec_extensions.json` 기준 enabled extension은 `14개`다.

- `차량교통량`
- `버스서비스강도`
- `가로등밀도`
- `보행등밀도`
- `바닥형보행신호등설치밀도`
- `강수량`
- `기온`
- `고령인구비율`
- `아동인구비율`
- `공원접근성`
- `편의점밀도`
- `음식점밀도`
- `경제활동인구`
- `비경제활동인구`

disabled extension은 `5개`다.

- `보행자신호등밀도`
- `관내이동생활인구`
- `대도시권유입생활인구`
- `장기외국인생활인구`
- `단기외국인생활인구`

## 7. 현재 학습 구조

현재 최신 학습 스크립트 기준:

- 입력: 최근 `24개월`
- 출력: 다음 `1개월`
- 모델: `2-layer ConvLSTM`
- hidden dim: `24`
- optimizer: `AdamW`
- epochs: `30`
- early stopping patience: `7`
- loss:
  - weighted BCE
  - focal 성분
  - OHEM 형태 hard negative mining
- gradient clipping 적용
- WeightedRandomSampler 적용

### checkpoint 선택 기준

예전에는 validation precision/F0.5 위주였다.  
현재는 `validation에서 threshold 0.10~0.90 탐색` 후:

- `recall >= 0.70` 후보 우선
- 그 안에서 `F1 최대`
- 동률이면 `precision 최대`

이 규칙으로 뽑은 validation selected F1 중심으로 checkpoint를 고른다.

즉, 지금 학습 기준은 운영 지표와 더 가깝게 바뀌어 있다.

## 8. 현재 미래 예측 구조

현재 미래 예측은 `recursive 1-step` 방식이다.

### 흐름

1. 2024년 말까지의 실제 월 패널을 사용
2. 최근 24개월을 입력으로 2025-01 예측
3. 예측된 2025-01 확률을 다음 달 입력 history에 반영
4. 다시 24개월 창을 잡아 2025-02 예측
5. 이를 2027-12까지 반복

### 미래 exogenous 처리

미래 월의 외생 변수는 아래 규칙으로 템플릿화한다.

- 같은 달의 최근 3개년 중앙값 우선
- 없으면 최근 12개월 중앙값 fallback
- 그래도 없으면 최근 관측값 fallback

이전의 단순 복사보다 drift가 줄어들도록 바꿔둔 상태다.

## 9. 현재 평가 구조

### 월별 평가

공식 평가는 월별 validation/test 확률 기준이다.

- threshold 탐색: `0.10 ~ 0.90`, `0.01` 간격
- validation only로 threshold 선택
- test에서는 최종 평가만 수행

핵심 지표:

- Precision
- Recall
- F1
- PR-AUC
- ROC-AUC
- Brier
- Calibration curve

### calibration

월별 calibration 후보:

- `identity`
- `platt_logit_1d`
- `isotonic_1d`

연간 calibration 후보:

- `identity`
- `isotonic_1d`
- `platt_logit_1d`
- `platt_logit_2d`

주의:
- calibration은 무조건 Brier만 최소화하는 쪽으로 고르지 않는다.
- ranking을 심하게 망치는 calibrator는 덜 뽑히도록 제약을 추가해 둔 상태다.

## 10. 최근 성능 상태

최근 사용자가 확인한 대표 성능은 대략 아래 수준이었다.

- prevalence: `0.1034`
- precision: `0.4502`
- recall: `0.6974`
- f1: `0.5471`
- pr_auc: `0.5473`
- roc_auc: `0.9245`
- brier: `0.1036`
- baseline brier: `0.0927`

### 현재 해석

이 모델은 아직 production-ready가 아니다.

핵심 이유:

- PR-AUC가 아직 낮음
- precision이 낮아 false positive가 많음
- Brier가 baseline보다 낮지 않아 확률 자체는 아직 불안정함

즉:

- 순위화 능력은 랜덤보다는 낫지만 아직 약함
- threshold만 조정해서 해결되는 문제가 아님
- ranking, feature quality, calibration이 함께 약한 상태

현재 평가는 `개선 필요`가 맞다.

## 11. 이번에 이미 반영한 개선

이 문서 작성 시점 기준으로 아래 개선은 이미 코드에 반영되어 있다.

- 12개월 동시 예측에서 `1개월 예측`으로 변경
- 24개월 입력으로 변경
- spatial hotspot feature 추가
- exposure-normalized feature 추가
- 최근/이전 구간 차이 feature 추가
- weather / citywide activity variable 재포함
- 2-layer ConvLSTM 구조로 변경
- hard negative mining 강화
- weighted sampler 추가
- AdamW + gradient clipping 적용
- monthly / annual calibration 선택 로직 정교화

## 12. 지금 남아 있는 핵심 과제

아직 실무 수준으로 끌어올리려면 아래 과제가 남아 있다.

### 가장 우선

1. 새 구조로 코랩에서 다시 end-to-end 실행
2. `셀 5`의 고정 feature count assert 제거
3. 새 결과로 PR-AUC / Brier / precision 재확인

### 다음 우선순위

1. true feature-level error analysis
   - false positive grid 공통 패턴
   - false negative grid 공통 패턴

2. segment-wise calibration
   - 자치구/중심지/주거지 등 구간별 calibration 차이 점검

3. threshold 운영 이원화
   - 탐지용 threshold
   - 정책배치용 top-k / 정책용 threshold

4. 추가 feature source
   - 관내이동생활인구
   - 대도시권유입생활인구
   - 장기외국인생활인구
   - 단기외국인생활인구
   - 보행자신호등밀도

## 13. 코랩 재실행 순서

최신 파이프라인을 다시 검증하려면 다음 순서를 권장한다.

1. 최신 업로드 폴더 파일을 `/content`에 업로드
2. `셀 4`: build 실행
3. `셀 5`: build_summary / feature_count 출력만 확인
4. `셀 6`: train 실행
5. `셀 7`: 학습 로그 확인
6. `셀 8`: forecast 실행
7. `셀 11`: evaluate 실행
8. `셀 12`: HTML 지도 생성

중요:
- 예전 노트북 셀 중 `feature_count == 33` 또는 `29` 같은 assert는 제거해야 한다.
- 현재는 resolved feature 결과를 기준으로 봐야 한다.

## 14. 오래된 문서/가정

아래는 현재 기준으로 그대로 믿으면 안 되는 항목이다.

- `29개 feature` 고정
- `33개 feature` 고정
- `다음 12개월 동시 예측`이 최신 구조라는 가정
- `강수량/기온/경제활동인구/비경제활동인구`가 항상 비활성이라는 가정
- `셀 8/11/12`의 예전 inline 코드가 최신이라는 가정

특히 [handoff.md](/Users/a0067/Documents/Playground/이거넣어 코랩/handoff.md) 와 [데이터셋_설명_20260413.md](/Users/a0067/Documents/Playground/이거넣어 코랩/데이터셋_설명_20260413.md) 는 역사 참고용으로는 유용하지만, 현재 최신 모델 구조는 이 문서가 더 정확하다.

## 15. GitHub에 올릴 때 권장사항

로컬 AI가 이어서 작업하게 하려면 최소한 아래는 같이 버전 관리하는 것을 권장한다.

- 이 문서
- `진짜 이거 넣어 코랩` 폴더의 최신 스크립트 4개
- `feature_spec_extensions.json`
- 셀 코드 md 문서들

용량 문제 때문에 아래는 Git LFS 또는 별도 스토리지 권장:

- 대형 CSV
- GPKG
- npy 중간 산출물

즉, GitHub repo는 아래 2층 구조가 가장 좋다.

- 코드/문서: Git
- 대용량 데이터: LFS 또는 외부 저장소

## 16. 로컬 AI에게 바로 주면 좋은 프롬프트

다른 로컬 AI에게 넘길 때는 아래처럼 말하면 문맥을 빠르게 잡는다.

`LOCAL_AI_HANDOFF_20260414.md를 먼저 읽고, 업로드 폴더 최신 스크립트 기준으로 24개월 입력 -> 다음 1개월 예측 -> 2025~2027 연집계 ConvLSTM 파이프라인을 이어서 개선해줘. feature_count 고정 가정은 버리고, build_summary.json의 실제 resolved feature 기준으로 판단해줘. 목표는 PR-AUC, Precision, Brier를 함께 개선하는 것이다.`

## 17. 한 줄 요약

현재 프로젝트의 최신 정답 경로는 `24개월 입력 -> 다음 1개월 ConvLSTM 예측 -> 2025~2027 recursive 월예측 -> 연집계 -> calibration -> 지도 시각화` 이며, 아직 실무 수준은 아니지만 그 방향으로 재구성은 완료된 상태다.
