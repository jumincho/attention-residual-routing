# AttnRes Routing Research 보존 최종 보고서

## 1. 이 프로젝트가 무엇이었는가

이 저장소의 핵심 질문은 하나였습니다.

디코더 언어모델 내부의 Attention Residual을 이용해, 품질을 크게 잃지 않으면서 깊이를 적응적으로 건너뛸 수 있는 내부 routing signal을 만들 수 있는가?

초기 가설은 비교적 강했습니다.

- Attention Residual이 시퀀스별 depth demand를 드러낼 수 있다.
- 그 신호를 이용하면 입력마다 다른 skip 결정을 내릴 수 있다.
- 그러면 품질 대비 계산량, 나아가 실제 end-to-end latency까지 개선할 수 있다.

프로젝트 종료 시점의 결론은 이보다 훨씬 좁아졌습니다.

- 신호 자체는 실재한다.
- 넓은 범위의 prompt-fixed routing 승리는 살아남지 못했다.
- 대신 `cc_news`에서만 유지되는 좁은 quality-positive lane은 끝까지 남았다.
- 그 lane도 실제 deployment speed win으로 이어지지는 않았다.

## 2. 이 아카이브에 무엇을 남겼는가

이번 보존본에는 프로젝트의 구조와 실행 논리를 이해하는 데 필요한 코드만 남겼습니다.

- `src/attnres_routing/`: 모델, routing, 데이터, 분석, 학습, manifest 생성의 핵심 라이브러리
- `scripts/`: 실험 실행기, 평가기, 샤딩 도구, 집계 스크립트, 파이프라인 래퍼
- `configs/`: 주요 실험 라운드의 YAML 설정
- `README.md`
- `requirements.txt`
- 영문/한글 최종 보고서

의도적으로 제외한 항목은 다음과 같습니다.

- `.git/`
- `.venv/`
- `data/`
- `results/`
- `logs/`
- `external/`

사용자 요청이 "필요한 코드 파일만 남긴 보존본"이었기 때문에, 대용량 산출물과 환경물은 모두 제외했습니다.

## 3. 처음 보는 사람을 위한 코드 지도

빠르게 구조를 이해하려면 아래 파일들부터 보면 됩니다.

- `src/attnres_routing/model.py`: 디코더 LM과 AttnRes 관련 모델 구성
- `src/attnres_routing/routing.py`: routing 및 mask 선택 로직
- `src/attnres_routing/analysis.py`: 요약 지표와 bootstrap 유틸리티
- `src/attnres_routing/data.py`: 데이터셋 준비
- `src/attnres_routing/sequence_manifest.py`: train/dev/lockbox manifest 생성
- `scripts/train_lm.py`: 기본 학습 진입점
- `scripts/evaluate_functional_oracles.py`: oracle 평가
- `scripts/evaluate_prompt_routing.py`: 초기 prompt-routing 평가
- `scripts/train_candidate_conditioned_ranker_v7.py`: 후기 candidate-conditioned selector 학습
- `scripts/evaluate_deployment_measurement_v7.py`: 품질 및 지연시간 배포 평가
- `scripts/build_lockbox_manifests_v9.py`: V9 fresh split 생성

마지막까지 남겨 둔 운영상 수정 사항은 두 군데입니다.

- `scripts/evaluate_deployment_measurement_v7.py`
- `scripts/run_compare_v8_seed_repro_v9.sh`

## 4. 연구가 어떻게 진행되었는가

### Phase A. 초기 재현과 신호 발견

프로젝트는 작은 설정, 특히 WikiText 계열에서 faithful Block AttnRes 재현과 초기 prompt-routing 실험으로 시작했습니다.

이 단계에서 확인된 사실은 다음과 같습니다.

- Attention Residual에는 실제 depth 관련 신호가 있다.
- leave-one-block-out 계열 지표로 그 신호를 측정할 수 있다.
- 하지만 prompt-fixed sequence routing은 강한 global static skip baseline을 이기지 못했다.

즉, 최초 가설은 여기서 한 번 크게 수정되었습니다.

### Phase B. 통제 실험과 음의 결과 정리

이후 transfer-integrity controls, 더 긴 4-GPU trajectory run, batched functional-oracle 평가, stability-gated routing 평가가 추가되었습니다.

이 단계의 핵심 결론은 이미 명확했습니다.

- AttnRes는 의미 있는 정보를 담고 있다.
- 그러나 당시 routing policy는 그 정보를 실제 routing win으로 바꾸지 못했다.

이 시점의 상태는 원래 저장소의 `docs/current_status_v2.md`에 정리되어 있었습니다.

### Phase C. `cc_news` 확장과 candidate-conditioned routing 전환

프로젝트는 이후 실패한 prompt-fixed 접근에서 벗어나, 더 큰 `24x512` 모델과 `cc_news`를 대상으로 candidate-conditioned selector 계열로 중심을 옮겼습니다.

이 전환이 프로젝트를 끝까지 살린 핵심 변화였습니다.

후기 실험에서는 다음을 함께 다뤘습니다.

- candidate-conditioned selector
- richer feature family
- bank-based routing
- readiness 및 checkpoint selection
- lockbox split protocol
- systems-aware deployment 측정

### Phase D. V8 강화 라운드

V8은 살아남은 `cc_news` 메인 라인을 굳히는 라운드였습니다.

V8에서 확인된 점:

- 메인 라인이 fresh `final_A/B/C` lockbox에서 살아남았다.
- seed `42/43/44`에 대해 multiseed replication이 되었다.
- necessity 분석에서 matched standard hidden-only control은 같은 win을 재현하지 못했다.
- deployment-aware 평가에서도 wall-clock speed win은 나오지 않았다.

즉 V8은 넓은 승리가 아니라, 좁은 주장을 지지했습니다.

AttnRes 기반 candidate-conditioned routing에는 `cc_news`에서만 재현되는 저예산 quality-positive lane이 있다. 그러나 그것은 일반적 adaptive-routing 승리가 아니다.

### Phase E. V8 포렌식

V8은 중간에 끊긴 실행 이력이 있어 provenance가 약한 부분이 있었습니다. 그래서 V8 숫자가 수치적으로나 절차적으로 믿을 만한지 별도 포렌식이 진행되었습니다.

포렌식 결론은 다음과 같았습니다.

- V8 요약 수치는 raw artifact로부터 수치적으로 재현 가능했다.
- lockbox split 자체는 clean하고 disjoint해 보였다.
- 다만 freeze logging과 interrupted-run provenance 문서는 충분히 강하지 않았다.

그래서 fresh V9 lockbox가 필요했습니다.

### Phase F. V9 fresh lockbox 마감

V9는 최종 마감 라운드였습니다.

새로운 `dev_select_v9`와 열리지 않은 `final_D/E/F` lockbox를 만든 뒤, dev만 보고 winner를 고정하고 세 final split을 모두 평가했습니다.

V9 완주 시각은 `2026-03-28 13:39:41 KST`입니다.

동결된 winner는 다음과 같습니다.

| seed | step | bank | feature | selector |
| --- | ---: | ---: | --- | --- |
| 42 | 5500 | 32 | `attnres` | `rf_pair` |
| 43 | 6000 | 32 | `stp_scalar` | `hgb_pair` |
| 44 | 3500 | 32 | `attnres` | `rf_pair` |

## 5. 최종적으로 무엇을 알아냈는가

### 핵심 과학적 결론

최종적으로 가장 강하게 방어 가능한 문장은 다음입니다.

`cc_news`에는 재현 가능한 좁은 quality-positive routing lane이 있다.

조금 더 구체적으로 말하면:

- V9 frozen winner는 `final_D/E/F` 세 locked split 모두에서 static보다 낮은 손실을 유지했다.
- 9개 seed-split 평가 전체 평균 locked `delta_to_static`은 `-0.02339`였다.
- 평균 `fraction_improved`는 `0.16800`이었다.

### 끝내 살아남지 못한 주장

프로젝트는 출발점의 더 강한 주장을 지지하지 못합니다.

지지하지 못하는 내용:

- corpus 전반의 일반성
- 일반적인 prompt-fixed routing 승리
- oracle bank에 가까운 저-regret 근사
- static 대비 deployment speed win

### 배포 관점 결론

배포 단계가 끝까지 병목이었습니다.

V9 locked deployment 평균은 다음과 같습니다.

- mean deploy `delta_to_global_static`: `-0.01643`
- mean dynamic end-to-end seconds/sequence: `0.22158`
- mean latency delta vs static: `+0.03174 s/sequence`

즉 품질은 static보다 약간 좋았지만, 동적 정책은 static보다 더 느렸습니다.

## 6. 최종 해석

이 프로젝트가 실제로 발견한 것은 분명히 있습니다.

- Attention Residual은 내부 routing signal을 담을 수 있다.
- 후기 `cc_news` 설정에서는 그 신호로 좁은 quality-positive selector family를 만들 수 있다.

하지만 더 큰 꿈은 끝내 닫히지 않았습니다.

- 결과는 좁고, 일반적이지 않다.
- 품질 이득은 크지 않다.
- regret는 여전히 oracle upper bound와 거리가 멀다.
- deployment latency는 좋아지지 않는다.

따라서 논문화 가능한 가장 정직한 한 줄은 다음입니다.

AttnRes 기반 내부 routing signal은 `cc_news`에서 좁지만 재현 가능한 quality-positive lane을 만들 수 있지만, 아직 deployment speed win으로 이어지지는 않는다.

## 7. 최종 라운드 운영 메모

V9 말미에 실제로 수정된 엔지니어링 이슈는 다음입니다.

- `scripts/evaluate_deployment_measurement_v7.py`
  `stp_diff_components` 전달 누락으로 deployment 측정이 크래시하던 문제 수정
- `scripts/run_compare_v8_seed_repro_v9.sh`
  compare template 기본 문자열이 깨지던 문제 수정

후반에는 저장공간 문제도 있었습니다.

- 대용량 결과 디렉터리를 `/var/tmp/chojm-attnres-routing-research-results/`로 옮겼고
- `results/` 아래에는 symlink를 남겨 파이프라인을 계속 살렸습니다.

이 산출물들은 이번 코드 보존본에 포함하지 않았습니다.

## 8. 새로 여는 사람에게 권하는 읽기 순서

처음 보는 사람이 가장 빠르게 이해하려면 아래 순서를 권합니다.

1. 이 보고서를 먼저 읽는다.
2. `README.md`를 읽는다.
3. `src/attnres_routing/`를 본다.
4. `scripts/`를 본다.
5. 후기 설정은 `configs/scale_heterogeneity_v8/`와 `configs/scale_heterogeneity_v9/`를 중심으로 본다.

## 9. 한 줄 요약

이 프로젝트는 혼합된 결과로 남는 것이 정확합니다.

- 실제 내부 routing signal은 찾았다.
- 여러 넓은 희망은 스스로 반증했다.
- 대신 하나의 좁은 재현 가능한 quality-positive lane은 끝까지 남겼다.
- 그러나 그것을 설득력 있는 systems win으로 바꾸는 데는 실패했다.

이것이 이 연구의 가장 정확한 종료 상태입니다.
