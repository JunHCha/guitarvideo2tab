# Architecture Decision Records (ADR)

`docs/domain/decisions/` 디렉토리는 guitarvideo2tab 프로젝트의 주요 아키텍처 결정을 기록합니다.
코드만 봐서는 재현 불가능한 **"왜 이 선택을 했는가"**를 보존하는 것이 목적입니다.

## ADR 목록

| ID | 상태 | 제목 | 날짜 |
|----|------|------|------|
| [001](001-multimodal-late-fusion-pipeline.md) | Accepted | 멀티모달 Late Fusion 파이프라인 채택 | 2026-05-08 |

## 상태값

| 상태 | 의미 |
|------|------|
| `Proposed` | 논의 중, 아직 확정 전 |
| `Accepted` | 채택되어 실행 중 |
| `Deprecated` | 더 이상 사용하지 않으나 대체재 없음 |
| `Superseded by ADR-NNN` | 다른 ADR로 대체됨 |

## 새 ADR 추가

`/domain-docs adr new <title>` 명령으로 자동 추가하거나, 다음 템플릿을 사용:

```markdown
---
id: "NNN"
title: "..."
status: Proposed
date: "YYYY-MM-DD"
deciders: [...]
supersedes: null
---

## Context
## Decision
## Consequences
## Alternatives
## References
```

## 작성 원칙

1. **Context에 제약/요구사항 명시** — 단순한 배경이 아니라 결정의 "강제 조건"
2. **Alternatives 최소 2개 이상** — 비교 없는 결정은 ADR이 아님
3. **Consequences는 positive/negative 모두** — 트레이드오프를 숨기지 말 것
4. **ADR은 불변** — 내용 수정 대신 새 ADR로 supersede
5. **관련 코드/PR/이슈 링크** — References 섹션에 명시
