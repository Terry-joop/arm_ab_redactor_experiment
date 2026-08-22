# 사전등록 프로토콜 — 의료축 P0/P1

## 불변 조건

1. 공통 1층은 `scripts/dataset_builders/_lawmask_l1.py` 그대로 import한다.
2. 최종 span은 `L1 ∪ L2`이다. 어떤 2층도 1층 span을 제거할 수 없다.
3. Qwen teacher 프롬프트에는 **원문 words와 법 범주 정의만** 넣는다. L1/GLiNER/v1.4 라벨은 넣지 않는다.
4. Student는 Qwen 라벨 train split만 보고, H1–H5의 gold로 Qwen 라벨을 쓰지 않는다.
5. 모든 결과 JSON에는 L1 `definition_dict(floor_sha)`, teacher model, prompt SHA-256,
   모델 revision, threshold, commit SHA를 저장한다.

## P1: 100건 teacher 파일럿

의료 6개 데이터셋(Drug Reviews, Symptom2Dx, ADR, RedditMH, MedNLI, Mental Health)에서
각 16–17건 계통 표본을 뽑는다. Qwen3-32B에게 다음 8개 범주만 물어본다.

`disease`, `symptom`, `medication`, `medical treatment`, `mental health condition`,
`substance use`, `injury`, `reproductive health`.

출력은 `범주 :: 원문에 있는 연속 phrase` 목록이어야 한다. 코드는 phrase를 원래 `words`에
정확 정렬하지 못하면 **거절**하고 별도 JSONL에 기록한다. 생성한 오프셋을 그대로 믿지 않는다.

사람은 최소 50건에서 다음만 체크한다.

- span이 실제 문장에 있는가
- 8개 건강 범주에 해당하는가
- 빠진 핵심 건강 span이 있는가
- 불필요하게 일반어를 가렸는가

이 결과는 teacher 품질 보고용이다. 평가 gold로 승격하지 않는다.

## P2: A/B 본 평가

P1 통과 뒤에만:

1. **Arm A**: 같은 Qwen 프롬프트를 런타임 2층으로 사용한다.
2. **Arm B**: 같은 프롬프트로 train만 라벨링하고 ELECTRA-small token classifier를 학습한다.
3. 둘 모두 L1과 OR하고 H1/H3/H4/H5를 수행한다.
4. H1/H3/H4/H5는 L1-only, 현행 GLiNER 결합팔도 함께 한 JSON에 기록한다.

H2 TAB은 의료 teacher로는 만들 수 없다. PII 범주용 별도 Qwen prompt·라벨셋을 만든 뒤
Arm A/B에 동일하게 적용한다. H2 없이 “전체 2층 대체”라고 결론 내리지 않는다.
