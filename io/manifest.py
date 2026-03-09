from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Dict, Optional


@dataclass
class RunManifest:
    dataset_dir: str
    ckpt_dir: Optional[str] = None
    stats_path: Optional[str] = None
    metadata: Dict[str, object] | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

    @staticmethod
    def from_json(text: str) -> "RunManifest":
        data = json.loads(text)
        return RunManifest(**data)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @staticmethod
    def load(path: str) -> "RunManifest":
        with open(path, "r", encoding="utf-8") as f:
            return RunManifest.from_json(f.read())

