# Arm A/B 의료 2층 Redactor 실험

공통 L1(결정적 규칙) 위에서 두 2층을 비교하는 실험의 코드·프로토콜·결과 요약이다.

- **Arm A**: Qwen3-32B를 런타임 2층으로 직접 사용
- **Arm B**: Qwen3-32B teacher 라벨로 ELECTRA-small을 fine-tuning해 로컬 2층으로 사용
- 최종 출력은 두 경우 모두 `L1 ∪ L2`이다.

## 포함 / 제외

이 Git 폴더에는 코드, 프로토콜, 집계 결과 JSON만 포함한다. 의료 원문, Qwen teacher JSONL, 학습 split, ELECTRA weight, Qwen3-32B weight는 민감 텍스트·용량 문제 때문에 포함하지 않는다.

로컬 전체 작업본은 `/home/jovyan/redactor_arm_ab_experiment`에 있다. 이 공개 폴더는 그 작업본의 재현·검토용 경량 스냅샷이다.

## 현재 결과

H1 문맥 구분 420문장, validation F2로 선택한 Arm B threshold 0.51 기준:

| 방식 | Recall | 비의료 flag | 균형점수 |
|---|---:|---:|---:|
| Arm A · Qwen3-32B | 0.751 | 0.080 | 0.836 |
| Arm B · ELECTRA-small | 0.864 | 0.466 | 0.699 |

H1은 내부 문맥 probe이며 human-gold가 아니다. H3/H4/H5의 독립 평가 결과는 아직 생성 중이다.

`results/medical_evaluation.json`은 Qwen teacher hold-out에 대한 Arm B 모방 성능이며, 사람 정답 정확도가 아니다.

## 구성

- `PROTOCOL.md`: Arm A/B 공통 평가 설계
- `src/`: Qwen teacher와 H1 평가 스크립트
- `results/`: 공개 가능한 집계 수치

세부 시각화는 [GitHub Pages 대시보드](https://terry-joop.github.io/token_redaction_probe/arm-ab/)에서 확인한다.
