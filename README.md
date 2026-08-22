# Arm A/B 의료 2층 Redactor 실험

공통 L1(결정적 규칙) 위에서 두 2층을 비교하는 실험의 코드·프로토콜·결과 요약이다.

- **Arm A**: Qwen3-32B를 런타임 2층으로 직접 사용
- **Arm B**: Qwen3-32B teacher 라벨로 ELECTRA-small을 fine-tuning해 로컬 2층으로 사용
- 최종 출력은 두 경우 모두 `L1 ∪ L2`이다.

## 포함 / 제외

이 Git 폴더에는 코드, 프로토콜, 집계 결과 JSON만 포함한다. 의료 원문, Qwen teacher JSONL, 학습 split, ELECTRA weight, Qwen3-32B weight는 민감 텍스트·용량 문제 때문에 포함하지 않는다.

로컬 전체 작업본은 `/home/jovyan/redactor_arm_ab_experiment`에 있다. 이 공개 폴더는 그 작업본의 재현·검토용 경량 스냅샷이다.

## 현재 결과

현재 실제 독립 평가는 H1·H3·H5까지 완료했다.

| 하네스 | 현행 GLiNER | Arm A · Qwen3-32B | Arm B · ELECTRA-small |
|---|---:|---:|---:|
| H1 문맥 균형점수 ↑ | 0.802 | **0.836** | 0.699 |
| H3 MedMentions F1 ↑ | **0.622** | 0.601 | 0.484 |
| H3 MedMentions Recall ↑ | 0.797 | 0.686 | **0.799** (th=0.51) |
| Arm B 예산일치 th=0.80 H3 F1 / R | — | — | 0.552 / 0.731 |
| H5 회귀(필수 58/비가림 24) | 통과 | 비가림 2건 위반 | th=0.51: 1건 / th=0.80: 통과 |

H3은 공개 MedMentions 600문서의 핵심 14 UMLS TUI 문자 span을 쓰며, 직접적인 민감정보 human-gold는 아니다. 모든 L2는 120-word 청크 1,659개를 동일하게 본 뒤 원문 좌표로 합쳐 채점했다. Arm A의 생성 출력 중 91/1,659 청크(5.5%)는 원문 exact-alignment 검증을 통과하지 못해 L2 미탐지로 처리했다.

현재로서는 Arm A/B가 현행 GLiNER를 대체한다는 결론은 성립하지 않는다. 결과 JSON은 `results/h3_medmentions_arm_ab_chunked.json`, `results/h5_arm_ab.json`, `results/h3_medmentions_arm_b_budget_matched.json`, `results/h5_arm_b_budget_matched.json`에 있다. `results/medical_evaluation.json`은 Qwen teacher hold-out에 대한 Arm B 모방 성능이며, 사람 정답 정확도가 아니다.

## 구성

- `PROTOCOL.md`: Arm A/B 공통 평가 설계
- `src/`: Qwen teacher와 H1 평가 스크립트
- `results/`: 공개 가능한 집계 수치

세부 시각화는 [GitHub Pages 대시보드](https://terry-joop.github.io/token_redaction_probe/arm-ab/)에서 확인한다.
