# 어려운 질의의 발생 원인 주석 가이드라인

**guideline_version: v1.0 (동결 예정)**
대상: MS MARCO 1M hard decile 698개 질의 · 주석자 2인 독립 · 불일치만 제3자 판정

---

## 0. 무엇을 판단하는가

주석자는 질의 하나와 그에 대응하는 문서 10개를 보고, **이 열 개가 의미상 어떤
그룹 구조를 이루는지**를 판단한다.

판단하지 않는 것:

- 문서가 좋은 답인지 (관련성 평가 아님)
- 검색 시스템이 이 질의를 잘 처리했는지
- 문서들이 색인/클러스터 상에서 가까운지

**"클러스터에 같이 있나"를 묻지 않는다.** 오직 의미만 본다.

## 1. 보이는 것과 보이지 않는 것

**보인다**: 질의문, 문서 10개의 원문 passage, 문서 ID (임의 순서로 섞여 있음).

**보이지 않는다**: 그래프 구조, φ, 클러스터 ID, regions/biggest/groups,
hard/easy 여부, ANN 결과, GT 내 순위, qrels 여부, 기존 자동 유형 라벨.

문서 순서는 질의마다 다른 시드로 섞여 있다. **슬롯 1은 1순위가 아니다.**
순서에서 어떤 것도 추론하지 말 것.

## 2. 과업 A — 의미 그룹 분할

문서 10개를 다음 질문으로 묶는다.

> 이 문서들은 이 질의에 대해 **실질적으로 같은 정보 요구 또는 같은 의미 해석**에
> 답하는가?

- 같은 해석·같은 주제 → 같은 그룹
- 서로 다른 의미·개체·주제 → 다른 그룹
- 확신이 없는 문서 → `U`로 표시

출력은 슬롯 1~10 순서의 그룹 ID 열이다. 예: `A,A,A,B,B,C,D,E,F,G`

그룹 문자(A/B/C…)에는 순서나 중요도 의미가 없다. 어떤 문자를 쓰든 **어느 문서들이
함께 묶였는가**만 채점된다.

**과업 A를 먼저 끝내고 나서 과업 B로 간다.** 라벨을 먼저 정하면 그룹 분할이
라벨에 끌려간다.

## 3. 과업 B — 전체 발생 원인 라벨

| 라벨 | 정의 |
|---|---|
| `single_topic` | 대부분의 GT가 하나의 일관된 의미/주제를 이룸 |
| `genuine_polysemy` | 질의 자체에 둘 이상 자연스러운 의미·의도·개체 해석이 있고, 둘 이상이 복수 문서로 지지됨 |
| `surface_collision` | 문서들이 단어·철자·명칭 등 표면형만 공유할 뿐, 질의에 대한 일관된 다중 의미 해석을 이루지 않음 |
| `mixed` | 다의성과 표면형 충돌이 실질적으로 함께 존재해 하나로 못 고름 |
| `unclear` | 제공 정보만으로 신뢰성 있게 판단 불가 |

정확히 하나만 고른다.

## 4. 판정 규칙

1. **`genuine_polysemy`는 "문서가 여러 군데에 퍼져 있다"가 아니다.**
   *질의 표현 자체가* 여러 자연스러운 해석을 허용한다는 증거가 있어야 한다.
   질의문만 보고 "이 말은 A일 수도 B일 수도 있다"고 말할 수 있어야 한다.

2. **각 해석은 적어도 두 GT 문서에서 확인되는 것이 원칙이다.**
   해석이 하나뿐인 문서로만 지지되면 `surface_collision` 또는 `mixed`를 먼저 검토한다.

3. **단어가 겹쳐도 같은 개체·같은 질문에 답하면 `single_topic`일 수 있다.**
   표현이 달라도 같은 것을 말하면 같은 그룹이다.

4. **문서들이 모두 서로 다르다는 사실만으로 `surface_collision`을 붙이지 않는다.**
   하나의 넓은 주제를 각기 다른 측면에서 다루는 경우일 수 있다.

5. **큰 그룹이 하나 있다고 해서 `single_topic`이나 `genuine_polysemy`가 아니다.**
   그 큰 그룹이 질의의 자연스러운 해석이 아니라면 여전히 `surface_collision`이다
   (§5 예시 3).

6. **`unclear`는 실패가 아니라 유효한 결과다.** 억지로 분류하지 않는다.
   전문 지식이 없어 판단이 서지 않으면 `unclear` + `confidence: low`.

7. 규칙 1과 규칙 4가 동시에 걸리면 — 즉 다의성 증거도 있고 무관한 표면형 일치도
   실질적으로 있으면 — `mixed`.

## 5. Worked examples

### 예시 1 — `what county is north fork in` → `surface_collision`

```
s1  Suffolk County ... eastern end of Suffolk is split into two forks
s2  East Grand Forks, MN ... Polk County, Minnesota
s3  Northfork-area historical earthquake activity ... West Virginia
s4  American River ... confluence at Auburn
s5  North Fork is an unincorporated community in Madera County, California
s6  Weather in North Fork Shannon Canyon, California
s7  Prices Fork ... Montgomery County, Virginia
s8  North Canton ... Stark County, Ohio
s9  Northport ... Leelanau County, Michigan
s10 Ford County ... Kansas
```

그룹: 거의 전부 단독. `North Fork`라는 이름의 서로 다른 지명들, 그리고 `Forks`,
`Northport`, `Ford County` 등은 이름이 비슷할 뿐 같은 개체가 아니다.

`genuine_polysemy`가 아닌 이유: "north fork"가 여러 곳을 가리킬 수 있다는 것은
맞지만 (규칙 1은 통과할 여지가 있다), **어떤 해석도 두 문서 이상으로 지지되지
않는다** (규칙 2). 게다가 s8~s10은 "north fork"의 해석조차 아니고 철자만
비슷하다. → `surface_collision`.

### 예시 2 — `define: precipitous delivery` → `surface_collision`

```
s1  Precipitate delivery ... delivery after an unusually rapid labor   [산과]
s2,s3,s4,s5,s6,s7,s8  precipitation = 응결된 수증기가 떨어지는 것 / 물의 순환  [기상]
s9  precipitate = 갑자기 일어나는 것 (일반 정의)
s10 Delivery = 물건을 목적지로 운송하는 과정
```

그룹: `A`(기상 강수) 7개, `B`(산과적 급속분만) 1개, `C`(일반 사전정의) 1개,
`D`(배송) 1개.

**가장 큰 그룹이 7개인데도 `single_topic`이 아니다** — 질의 `precipitous
delivery`는 산과 용어이고, 기상학의 "precipitation"은 이 질의의 자연스러운 해석이
아니다. 어간만 공유한다 (규칙 5). `genuine_polysemy`도 아니다: 질의문 자체가
"급속분만"과 "강수"라는 두 해석을 자연스럽게 허용하지 않는다 (규칙 1).
→ `surface_collision`.

**이 예시가 largest-group-fraction과 라벨이 별개인 이유다.** 그룹 크기는 따로
연속량으로 보고되므로, 라벨에 그것을 반영하려 하지 말 것.

### 예시 3 (경계 사례) — `what is mhs?` → 주석자 판단, 불일치 시 판정 대상

```
s1  Protos ... 이름이 M과 S로 시작하는 두 이탈리아인
s2  Multiple Sclerosis (MS) ... 만성 신경질환
s3  Multiple sclerosis (MS) ... 중추신경계 질환
s4  melanocyte-stimulating hormone, Abbr. MSH
s5  HPE System Management Homepage (SMH)
s6  Specialist High Skills Major (SHSM)
s7  Medical Subject Headings (MeSH)
s8  Malignant hyperthermia (MH)
s9  Children's Special Health Care Services (CSHCS)
s10 Multi-Health Systems Inc. (MHS)
```

그룹: `MS`(s2,s3) 2개, 나머지는 전부 단독이고 **서로 다른 약어**다 — MSH, SMH,
SHSM, MeSH, MH, CSHCS.

두 방향의 읽기가 모두 가능하다:

- `genuine_polysemy`: "MHS"는 실제로 다의적 약어이고, 복수 문서로 지지되는 해석
  (multiple sclerosis)이 하나 존재한다.
- `surface_collision`: 질의어는 `MHS`인데 문서 대부분은 **MHS가 아닌 다른 약어**다
  (M/S/H 글자 조합만 겹친다). 실제로 MHS를 다루는 문서는 s10 하나뿐이므로 규칙 2에
  걸린다.

가이드라인은 이 사례를 **강제로 한쪽으로 밀지 않는다.** 규칙 2를 엄격히 적용하면
`surface_collision`, 약어 다의성을 우선하면 `mixed`. 자신의 판단을
`brief_rationale`에 근거와 함께 남기고, 불일치는 판정 단계에서 해소한다.

## 6. confidence와 brief_rationale

`confidence`:

- `high` — 다른 주석자도 같게 볼 것이라 본다
- `medium` — 근거는 있으나 다른 라벨도 방어 가능하다
- `low` — 사실상 추측이다 (`unclear`와 함께 쓰는 경우가 많다)

`brief_rationale`은 **1~2문장, 필수**다. 심사자가 "왜 이것이 surface collision인가"
라고 물었을 때 대표 사례를 재구성하고 오류를 감사하기 위한 것이다. 좋은 근거는
*무엇이 무엇과 겹치는지*를 말한다:

- 좋음: "문서 대부분이 서로 다른 지명이며, 어느 해석도 2개 이상 문서로 지지되지 않음."
- 나쁨: "문서들이 흩어져 있음." / "표면형 충돌임."

## 7. 출력 한 줄의 형식

```
query_id, query_text, passage_ids_shuffled, annotator_id,
semantic_group_ids,     # 슬롯 1~10 순서, 쉼표 구분. 예: A,A,A,B,B,C,D,E,F,G
primary_label,          # single_topic | genuine_polysemy | surface_collision | mixed | unclear
confidence,             # high | medium | low
brief_rationale,        # 1~2문장
timestamp, guideline_version
```

## 8. 운영과 동결

1. 무작위 50개 파일럿, 2인 독립 주석 (`v0.9-pilot`).
2. 불일치 사례를 모아 가이드라인과 예시를 보완한다.
3. **가이드라인을 `v1.0`으로 동결한다. 이후 결과를 보고 기준을 바꾸지 않는다.**
4. 파일럿 항목을 포함해 698개 전부를 2인 독립 주석 (`v1.0`).
5. 불일치 항목만 제3 주석자 또는 합의 회의로 판정.
6. k-means 휴리스틱 라벨은 **마지막에만** 결합해 사람 라벨과 비교한다.

동결 이후 가이드라인이 바뀌면 버전을 올리고, 그 버전으로 주석된 항목을 별도로
보고한다. 라벨을 조용히 갱신하지 않는다.

---

**폐지된 범주**: 기존 `weak senses`는 의미론적 원인이 아니라 "의미 그룹이 작다"는
구조적 관찰이므로 라벨에서 제외한다. 과업 A의 그룹 분할에서
`largest semantic-group fraction`으로 자동 계산되어 연속량으로 보고된다.
