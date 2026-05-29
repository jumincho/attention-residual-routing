<div align="center">

# attention-residual-routing

**어텐션 잔차로 입력별 깊이 라우팅을 학습할 수 있는가**
**Can attention-residual signal learn per-input depth routing?**

![Status](https://img.shields.io/badge/status-dormant-lightgrey)
![Language](https://img.shields.io/badge/language-Python-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-CC%20BY--NC%204.0-lightgrey)
![Closure](https://img.shields.io/badge/closure-2026--03-blue)

**한국어** · [English](#english) · [中文](./README.zh-CN.md)

</div>

> 🧊 **휴면(dormant) 중인 연구 파일럿입니다.**

### 빠른 시작

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m py_compile src/attnres_routing/*.py   # 설치 확인
```

## 무엇을 보려던 연구였나

언어모델 안의 여러 층(layer)이 모든 입력에 대해 항상 똑같이 일을 하는 건 아닙니다. 어떤 문장은 깊은 층까지 가야 풀리고, 어떤 문장은 얕은 층에서 끝나도 됩니다. 그렇다면 **모델 내부의 어떤 신호를 보고 "이 입력에는 깊은 층이 필요한지" 를 판단해, 불필요할 때는 일부 층을 건너뛰게 할 수 있을까** — 가 출발 질문이었습니다.

핵심 가설은 셋이었습니다.

- 모델 내부의 어떤 신호(어텐션 계열 잔차)는 입력별로 "깊이가 얼마나 필요한지" 를 드러낸다
- 그 신호를 보고 입력마다 다른 건너뛰기 결정을 내릴 수 있다
- 결과적으로 같은 품질을 유지하면서 계산량과 실제 추론 시간까지 줄일 수 있다

스몰 스케일에서 시작해 점차 더 큰 모델, 더 다양한 데이터셋(WikiText, TinyStories, OpenWebText, FineWeb-Edu, CC News 등)으로 확장하며 검증했습니다.

## 무엇을 알아냈나

- **그 신호 자체는 실재했습니다.** 깊이별 활용도를 측정하면 입력에 따라 의미 있게 다르게 나왔습니다.
- **하지만 처음에 노렸던 "어떤 데이터셋·어떤 설정에서도 통하는 일반적인 건너뛰기 정책" 은 살아남지 못했습니다.** 라우팅 정책으로 바꿔 보면, 단순히 항상 같은 층을 건너뛰는 정적 baseline 을 못 이기는 경우가 많았습니다.
- **그래도 끝까지 살아남은 좁은 결과 하나**가 있었습니다. 한 가지 특정 데이터셋(`cc_news`) 에 한해, 여러 시드에 걸쳐 재현되는 작은 품질 우위가 남았습니다. 새 데이터 분할(lockbox)에서도 그 좁은 우위는 살아남았습니다.
- **다만 실제 추론 속도까지 빨라지지는 않았습니다.** 품질은 조금 더 좋아도, 동적으로 건너뛰는 비용 때문에 wall-clock 으로는 정적보다 느렸습니다.

자세한 결과가 궁금하시면:

- 🇰🇷 [`reports/PROJECT_FINAL_REPORT_KO.md`](reports/PROJECT_FINAL_REPORT_KO.md)
- 🇬🇧 [`reports/PROJECT_FINAL_REPORT_EN.md`](reports/PROJECT_FINAL_REPORT_EN.md)

## 왜 잠시 멈춰 두는가

신호는 분명히 있고, 좁은 영역에서 재현 가능한 품질 우위까지 남겼습니다. 다만 그것을 **실제로 빠른 시스템** 으로 바꾸는 데는 실패했고, 더 넓은 일반성(다른 코퍼스로의 전이)도 잡지 못했습니다. 다음 자극(다른 데이터, 더 가벼운 selector, 다른 라우팅 기하)이 생기면 다시 깨우는 편이 자연스럽다고 판단했습니다.

## 다시 들여다볼 때는 어디부터

- 📖 [`GLOSSARY.md`](GLOSSARY.md) — 소스·설정·최종 보고서에 그대로 살아남은 내부 용어(데이터셋 별칭, 서브레이어 마스크 표기, 라우팅 점수 모드, `_v5`~`_v9` 라운드, lockbox 등)를 풀어 둔 문서
- 🇰🇷 [`reports/PROJECT_FINAL_REPORT_KO.md`](reports/PROJECT_FINAL_REPORT_KO.md) — 한 편 분량의 최종 보고서
- [`src/attnres_routing/`](src/attnres_routing/) — 모델, 라우팅, 데이터, 분석의 핵심 라이브러리
- 후기 설정 폴더 안의 YAML 들이 마지막 단계의 실험 형태를 가장 잘 보여 줍니다

## 코드 어디에 뭐가 있나

| 파일 | 하는 일 |
|---|---|
| [`src/attnres_routing/model.py`](src/attnres_routing/model.py) | 디코더 모델과 어텐션 잔차 부분의 구성 |
| [`src/attnres_routing/routing.py`](src/attnres_routing/routing.py) | 어느 층을 건너뛸지 고르는 라우팅 로직 |
| [`src/attnres_routing/analysis.py`](src/attnres_routing/analysis.py) | 요약 지표와 통계 |
| [`src/attnres_routing/data.py`](src/attnres_routing/data.py) | 데이터셋 준비 |
| [`src/attnres_routing/sequence_manifest.py`](src/attnres_routing/sequence_manifest.py) | 학습/검증/잠금(lockbox) 분할 매니페스트 생성 |
| [`src/attnres_routing/sublayer_masks.py`](src/attnres_routing/sublayer_masks.py) | `SublayerMask`, `action_types`, `to_id()` 인코딩, 후보 마스크 열거, FLOP 추정 |
| [`src/attnres_routing/normalizers.py`](src/attnres_routing/normalizers.py) | 깊이축 정규화 함수(`softmax` / `sparsemax` / `entmax15` / `topk_softmax`) |
| [`src/attnres_routing/train.py`](src/attnres_routing/train.py) | LM 사전학습 루프(DDP / AMP / cosine LR / STP 정규화) |
| [`src/attnres_routing/utils.py`](src/attnres_routing/utils.py) | 시드 고정, YAML I/O, 디렉터리 생성, HF token, cosine LR, 파라미터 카운트 |
| [`scripts/train_lm.py`](scripts/train_lm.py) | 기본 학습 진입점 |
| [`scripts/evaluate_functional_oracles.py`](scripts/evaluate_functional_oracles.py) | 이상적인 정책을 oracle 로 두고 평가 |
| [`scripts/evaluate_prompt_routing.py`](scripts/evaluate_prompt_routing.py) | 초기 형태의 입력별 라우팅 평가 |
| [`scripts/train_candidate_conditioned_ranker_v7.py`](scripts/train_candidate_conditioned_ranker_v7.py) | 후기에 메인이 된 후보 조건부 selector 학습 |
| [`scripts/evaluate_deployment_measurement_v7.py`](scripts/evaluate_deployment_measurement_v7.py) | 품질과 실제 지연 시간을 함께 측정 |
| [`scripts/build_lockbox_manifests_v9.py`](scripts/build_lockbox_manifests_v9.py) | 새로 본 적 없는 검증 분할 생성 |

위에 나열된 스크립트들은 핵심 진입점이며, `scripts/` 안의 나머지 파일들은 참고 보존을 위해 남겨 둔 v5–v9 실험 파이프라인 러너들입니다.

## 폴더 지도

```
.
├── src/attnres_routing/   모델 / 라우팅 / 데이터 / 분석 / 학습 / 매니페스트
├── scripts/               학습·평가·집계·파이프라인 진입점들
├── configs/               라운드별 실험 설정
├── reports/               최종 보고서 (한국어 / 영문)
├── GLOSSARY.md            내부 용어 해설
└── requirements.txt
```

대용량 산출물(데이터, 결과, 로그, 외부 의존)은 이 보존본에 포함하지 않았습니다.

## 환경

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export HF_TOKEN=...   # 필요한 경우에만
```

## 상태

🧊 **휴면 중** — 좁고 재현 가능한 품질 우위는 남았지만, 실제 속도 우위까지는 가지 못한 상태입니다.

---

<a name="english"></a>

## English

> 🧊 **Dormant research pilot.**

### Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m py_compile src/attnres_routing/*.py   # sanity check
```

### What this set out to test

Layers inside a language model do not all do the same amount of work on every input. Some sequences resolve in shallow layers; others need the deeper stack. The starting question was: **can we read some internal signal that says "this input needs depth" — and skip layers when it doesn't?**

Three core hypotheses:

- An internal signal (attention-style residuals) reveals per-input "how much depth is needed."
- That signal supports per-input skip decisions.
- The skipping cuts compute and real wall-clock latency while preserving quality.

Tested at small scale first, then scaled up to larger models and more diverse corpora (WikiText, TinyStories, OpenWebText, FineWeb-Edu, CC News, etc.).

### What it found

- **The signal itself is real.** Per-depth utilization measurably varies by input.
- **The general "skips that work across datasets" policy didn't hold.** When converted into a routing policy, it often failed to beat a static baseline that always skips the same layers.
- **One narrow result did survive.** On a specific corpus (`cc_news`), a small reproducible quality edge held across seeds and even on a fresh lockbox split.
- **But it did not translate into actual speed.** Quality improved slightly; wall-clock latency was worse than the static policy because of the dynamic-skip overhead.

Full results:

- 🇰🇷 [`reports/PROJECT_FINAL_REPORT_KO.md`](reports/PROJECT_FINAL_REPORT_KO.md)
- 🇬🇧 [`reports/PROJECT_FINAL_REPORT_EN.md`](reports/PROJECT_FINAL_REPORT_EN.md)

### Why it's on hold

The signal is real and produces a reproducible — though narrow — quality edge. What it doesn't do is become a **faster** system, and it doesn't transfer to other corpora. Waiting for a fresh angle (different data, lighter selector, different routing geometry) is the natural next step rather than continuing on the current path.

### Where to look first when revisiting

- 📖 [`GLOSSARY.md`](GLOSSARY.md) — decodes the internal vocabulary (dataset aliases, sublayer-mask format, routing score modes, `_v5`–`_v9` rounds, lockbox, candidate-conditioned ranker, etc.) used across source, configs, and reports.
- 🇬🇧 [`reports/PROJECT_FINAL_REPORT_EN.md`](reports/PROJECT_FINAL_REPORT_EN.md) — full final report.
- [`src/attnres_routing/`](src/attnres_routing/) — core library (model, routing, data, analysis).
- The YAMLs in the late-round config folders are the cleanest snapshot of the final experimental shape.

### Code map

| File | What it does |
|---|---|
| [`src/attnres_routing/model.py`](src/attnres_routing/model.py) | Decoder model + attention-residual hooks |
| [`src/attnres_routing/routing.py`](src/attnres_routing/routing.py) | Per-input layer-skip routing logic |
| [`src/attnres_routing/analysis.py`](src/attnres_routing/analysis.py) | Summary metrics and statistics |
| [`src/attnres_routing/data.py`](src/attnres_routing/data.py) | Dataset preparation |
| [`src/attnres_routing/sequence_manifest.py`](src/attnres_routing/sequence_manifest.py) | Train / val / lockbox split manifests |
| [`src/attnres_routing/sublayer_masks.py`](src/attnres_routing/sublayer_masks.py) | `SublayerMask`, `action_types`, `to_id()` encoding, candidate enumeration, FLOP estimation |
| [`src/attnres_routing/normalizers.py`](src/attnres_routing/normalizers.py) | Depth-axis normalizers (`softmax` / `sparsemax` / `entmax15` / `topk_softmax`) |
| [`src/attnres_routing/train.py`](src/attnres_routing/train.py) | LM pretraining loop (DDP / AMP / cosine LR / STP regularizer) |
| [`src/attnres_routing/utils.py`](src/attnres_routing/utils.py) | Seeding, YAML I/O, dir creation, HF token, cosine LR, parameter counting |
| [`scripts/train_lm.py`](scripts/train_lm.py) | Base training entrypoint |
| [`scripts/evaluate_functional_oracles.py`](scripts/evaluate_functional_oracles.py) | Evaluation against oracle policies |
| [`scripts/evaluate_prompt_routing.py`](scripts/evaluate_prompt_routing.py) | Early per-input routing evaluation |
| [`scripts/train_candidate_conditioned_ranker_v7.py`](scripts/train_candidate_conditioned_ranker_v7.py) | Late-round candidate-conditioned selector training |
| [`scripts/evaluate_deployment_measurement_v7.py`](scripts/evaluate_deployment_measurement_v7.py) | Joint quality + latency measurement |
| [`scripts/build_lockbox_manifests_v9.py`](scripts/build_lockbox_manifests_v9.py) | Unseen validation split construction |

The scripts listed above are the core entry points; the remaining files in `scripts/` are v5–v9 experiment pipeline runners kept for reference.

### Folder map

```
.
├── src/attnres_routing/   model / routing / data / analysis / training / manifests
├── scripts/               training, evaluation, aggregation entrypoints
├── configs/               per-round experiment configs
├── reports/               final reports (KO / EN)
├── GLOSSARY.md            internal vocabulary
└── requirements.txt
```

Large artifacts (datasets, results, logs, external dependencies) are not included in this archive.

### Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export HF_TOKEN=...   # only if needed
```

### Status

🧊 **Dormant** — a narrow reproducible quality edge survives; the speed win did not.

### License

Released under [CC BY-NC 4.0](./LICENSE).
