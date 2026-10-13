"""관절 순서와 개발용 연결을 명시적으로 관리한다."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Skeleton:
    document: dict
    sha256: str

    @property
    def groups(self) -> dict[str, list[str]]:
        return self.document["groups"]

    @property
    def joint_names(self) -> list[str]:
        return list(dict.fromkeys(n for names in self.groups.values() for n in names))


def load_skeleton(path: str | Path) -> Skeleton:
    raw = Path(path).read_bytes()
    doc = json.loads(raw)
    if not isinstance(doc, dict) or not all(doc.get(k) for k in ("id", "status", "evidence")):
        raise ValueError("Skeleton에는 id, status, evidence가 필요합니다.")
    groups = doc.get("groups")
    if not isinstance(groups, dict) or not groups:
        raise ValueError("Skeleton groups는 비어 있지 않은 사전이어야 합니다.")
    for group, names in groups.items():
        if not isinstance(names, list) or not names or any(not isinstance(n, str) or not n for n in names):
            raise ValueError(f"잘못된 관절 목록: {group}")
        if len(set(names)) != len(names):
            raise ValueError(f"그룹 내부 중복 관절: {group}")
    selected = {n for names in groups.values() for n in names}
    edges = doc.get("edges")
    if not isinstance(edges, list):
        raise ValueError("Skeleton edges는 목록이어야 합니다.")
    seen = set()
    for edge in edges:
        if not isinstance(edge, list) or len(edge) != 2 or any(not isinstance(n, str) for n in edge):
            raise ValueError(f"잘못된 연결: {edge}")
        a, b = edge
        if a == b or a not in selected or b not in selected or tuple(sorted(edge)) in seen:
            raise ValueError(f"미선택 관절/자기 연결/중복 연결: {edge}")
        seen.add(tuple(sorted(edge)))
    return Skeleton(doc, hashlib.sha256(raw).hexdigest())
