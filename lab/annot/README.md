# 698 hard queries, double-annotated

k-means 휴리스틱 라벨을 사람 전수 주석으로 대체하기 위한 도구 일체.
바꾸려는 이유는 하나다 — 기존 유형 비율은 질의의 성질이 아니라 **k의 함수**다
(surface collision이 k=256에서 31.8%, k=2048에서 52.4%). 사람은 색인을 보지 않고
질의와 문서 10개만 보고 판단하므로, 라벨이 설명 대상에서 되읽히지 않는다.

## 파일

| 파일 | 내용 | 주석자에게 |
|---|---|---|
| `GUIDELINES.md` | 가이드라인 v1.0, 판정 규칙, worked example 3개 | **준다** |
| `items_all.jsonl` | 698항목: 질의문 + 문서 10개(섞인 순서) + 슬롯 | **준다** |
| `items_pilot.jsonl` | 그중 무작위 50개 (파일럿) | **준다** |
| `annotate.html` | 오프라인 주석 도구 (브라우저에서 열기) | **준다** |
| `sheets/*.csv` | 도구를 안 쓸 경우의 빈 시트 | 준다 |
| `key_all.jsonl` | 슬롯→순위, qrels 여부, 휴리스틱 라벨, φ₁, recall | **절대 안 준다** |
| `batches.json` | 시드, 파일럿 질의 목록, 배치별 guideline_version | — |
| `dryrun/` | 가짜 주석. 분석 경로 점검용, **데이터 아님** | — |

`items_*.jsonl`에는 순위·qrels·클러스터·φ·hard 여부가 들어 있지 않다. 문서 순서는
질의별 시드로 섞여 있어 슬롯 1이 1순위가 아니다 (시드는 `shuffle_seed`에 기록).

## 절차

```bash
python code/annot_prep.py                    # 항목 + 봉인 키 생성 (재실행해도 동일)

# 1. 파일럿 50개, 2인 독립  (guideline v0.9-pilot)
python code/annot_agree.py --a1 annot/records_pilot_A1.jsonl \
                           --a2 annot/records_pilot_A2.jsonl
#    -> annot/to_adjudicate.jsonl 을 읽고 가이드라인과 예시를 보완

# 2. GUIDELINES.md 를 v1.0 으로 동결. 이후 결과를 보고 기준을 바꾸지 않는다.

# 3. 파일럿 포함 698개 전부, 2인 독립  (v1.0)
python code/annot_agree.py                   # 기본값이 records_main_A{1,2}.jsonl

# 4. 불일치만 제3자/합의로 판정 -> annot/records_adjudicated.jsonl
python code/annot_agree.py --phi3            # 최종 집계. 휴리스틱은 여기서 처음 결합
```

`annot_agree.py`는 라벨 오타, 그룹 개수(10개) 불일치, 빈 rationale, 배치를 넘나드는
`guideline_version`을 모두 잡아서 멈추거나 경고한다.

## 도구 사용

`annotate.html`을 브라우저에서 열고 → annotator id 입력 → `items…`로 항목 파일 선택.
작업 내용은 브라우저 localStorage에 자동 저장되고, `export`가
`records_<id>.jsonl`을 스펙 형식 그대로 내려준다. **export를 자주 할 것** —
저장은 그 브라우저 안에만 있다. 중단 후 이어서 하려면 `resume…`로 내려받은
파일을 다시 올린다.

10개 슬롯을 모두 묶고, 라벨·confidence·rationale을 채워야 `complete`가 된다.
미완성 항목은 export되지 않으므로 두 주석자의 항목 수가 곧 진행률이다.

## 논문에 나가는 것

`results/annot_summary.json` 하나에 다 들어간다.

1. 최종 사람 라벨 비율 + 95% Wilson 신뢰구간
2. 합의 전 라벨 일치도: Cohen's κ, Krippendorff's α
3. 그룹 분할 일치도: 문서쌍 pairwise F1 (macro/micro), ARI
4. 사람 라벨 × 휴리스틱 라벨 혼동행렬
5. 라벨별 φ₁, φ₃, recall@ef16, largest semantic-group fraction

> **κ 두 개가 다른 것이다.** 여기의 Cohen's κ는 주석자간 일치도이고, 논문에서
> 뺀 cohesion κ와 무관하다. 혼동을 피하려면 헤드라인은 Krippendorff's α로 쓰고
> κ는 괄호에 두는 편이 낫다.

`weak senses`는 라벨에서 폐지됐다. "의미 그룹이 작다"는 구조적 관찰이므로
과업 A의 그룹 분할에서 largest semantic-group fraction으로 자동 계산되어
연속량으로 보고된다. `regions`는 필요하면 k-means 보조 진단값으로만 남긴다.

## 알아둘 것

- 여기서 **GT 10개는 stock encoder의 exact top-10**이다 (`gt.py`, Ruler #1).
  qrels가 아니다. 즉 주석자는 "무엇이 정답인가"가 아니라 "인코더가 한데 모아 온
  열 개가 의미상 무엇인가"를 판단한다 — φ가 정의된 그 집합과 같은 집합이다.
- `msmarco-1m` 코퍼스는 passage 한 줄이 전부라 **제목/URL이 없다.** 스펙의
  "필요하면 제목/URL"은 이 코퍼스에서는 제공할 수 없다.
- 항목 생성은 결정적이다. `annot_prep.py`를 다시 돌려도 같은 셔플이 나오므로
  주석 도중 재생성해도 안전하다.
