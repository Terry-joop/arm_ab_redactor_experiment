# Arm A/B 2층 Redactor 실험

공통 L1(결정적 규칙) 위에서 두 2층을 비교하는 실험의 코드·프로토콜·결과 요약이다.

- **Arm A**: Qwen3-32B를 런타임 2층으로 직접 사용
- **Arm B**: Qwen3-32B teacher 라벨로 ELECTRA-small을 fine-tuning해 로컬 2층으로 사용
- 최종 출력은 두 경우 모두 `L1 ∪ L2`이다.

## 포함 / 제외

이 Git 폴더에는 코드, 프로토콜, 집계 결과 JSON만 포함한다. 의료 원문, Qwen teacher JSONL, 학습 split, ELECTRA weight, Qwen3-32B weight는 민감 텍스트·용량 문제 때문에 포함하지 않는다.

로컬 전체 작업본은 `/home/jovyan/redactor_arm_ab_experiment`에 있다. 이 공개 폴더는 그 작업본의 재현·검토용 경량 스냅샷이다.

## 현재 결과

의료축 H1·H3·H4·H5와, PII축 H2 Arm A 평가를 완료했다. H2 Arm B는 별도 PII Teacher
라벨과 학습 코퍼스가 필요해 아직 수행하지 않았다.

| 하네스 | 현행 GLiNER | Arm A · Qwen3-32B | Arm B · ELECTRA-small |
|---|---:|---:|---:|
| H1 문맥 균형점수 ↑ | 0.802 | **0.836** | 0.699 |
| H2 TAB PII F1 ↑ (ECHR 1,014문서) | **0.643** (web_md) | 0.567 | PII Student 미학습 |
| H3 MedMentions F1 ↑ | **0.622** | 0.601 | 0.484 |
| H3 MedMentions Recall ↑ | 0.797 | 0.686 | **0.799** (th=0.51) |
| Arm B 예산일치 th=0.80 H3 F1 / R | — | — | 0.552 / 0.731 |
| H5 회귀(필수 58/비가림 24) | 통과 | 비가림 2건 위반 | th=0.51: 1건 / th=0.80: 통과 |

H3은 공개 MedMentions 600문서의 핵심 14 UMLS TUI 문자 span을 쓰며, 직접적인 민감정보 human-gold는 아니다. 모든 L2는 120-word 청크 1,659개를 동일하게 본 뒤 원문 좌표로 합쳐 채점했다. Arm A의 생성 출력 중 91/1,659 청크(5.5%)는 원문 exact-alignment 검증을 통과하지 못해 L2 미탐지로 처리했다.

H2는 공개 TAB ECHR 1,014문서의 문자 단위 DIRECT/QUASI gold를 hold-out으로 사용한다. Arm A는 phrase 재탐색 대신 Qwen이 원문 단어의 zero-based index JSON을 출력하고, 범위·중복·JSON을 검증한 뒤에만 마스킹한다. 전체 11,849청크에서 Qwen 출력 검증 실패는 233건(2.0%)이었다. TAB은 Teacher 라벨 또는 Student 학습에 사용하지 않았다.

현재로서는 Arm A/B가 현행 2층을 대체한다는 결론은 성립하지 않는다. 결과 JSON은 `results/h2_tab_pii_arm_a_structured_full.json`, `results/h3_medmentions_arm_ab_chunked.json`, `results/h5_arm_ab.json`, `results/h5_arm_b_budget_matched.json`에 있다. `results/medical_evaluation.json`은 Qwen teacher hold-out에 대한 Arm B 모방 성능이며, 사람 정답 정확도가 아니다.

## 구성

- `PROTOCOL.md`: Arm A/B 공통 평가 설계
- `src/`: Qwen teacher와 H1 평가 스크립트
- `results/`: 공개 가능한 집계 수치

세부 시각화는 [GitHub Pages 대시보드](https://terry-joop.github.io/token_redaction_probe/arm-ab/)에서 확인한다.
