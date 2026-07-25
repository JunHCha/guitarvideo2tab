"""실행(run) 단위 산출물 워크스페이스.

trial 을 반복하면 입력·중간산출물·결과가 한 디렉토리에 섞여 어느 파일이 어느
실행에서 나왔는지 추적할 수 없다. 이 모듈은 **실행 하나당 디렉토리 하나**를
보장하고, 무엇을 어떤 설정으로 돌렸는지를 ``manifest.json`` 에 남긴다.

레이아웃::

    runs/
    ├── latest -> 20260725-234512_주이름-찬양-verse-1   (심볼릭 링크)
    └── 20260725-234512_주이름-찬양-verse-1/
        ├── manifest.json      실행 메타데이터 (입력·설정·커밋·단계별 통계)
        ├── input/             원본 입력 사본 또는 참조
        ├── stages/            단계별 중간 산출물 (미디어 + JSON dump)
        └── output/            최종 결과 (tab.mid / tab.musicxml)

``runs/`` 는 통째로 gitignore 되며 ``.gitkeep`` 만 추적된다.
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

RUNS_DIRNAME = "runs"
LATEST_LINK = "latest"

_RUN_ID_TIME_FORMAT = "%Y%m%d-%H%M%S"
_SLUG_MAX_LEN = 48

# 파일명에 쓸 수 없거나 셸에서 성가신 문자. 한글·영숫자·하이픈만 남긴다.
_SLUG_STRIP = re.compile(r"[^0-9A-Za-z가-힣]+")


def slugify(value: str) -> str:
    """입력 이름을 파일시스템에 안전한 슬러그로 바꾼다(한글 보존)."""
    stem = Path(value).stem if not value.startswith(("http://", "https://")) else value
    slug = _SLUG_STRIP.sub("-", stem).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)
    return slug[:_SLUG_MAX_LEN] or "run"


def current_git_commit(repo_dir: Path | None = None) -> str | None:
    """현재 커밋 SHA(짧은 형식). 저장소가 아니거나 git 이 없으면 None."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_dir) if repo_dir else None,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - 환경 의존
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


@dataclass
class RunWorkspace:
    """한 번의 파이프라인 실행이 쓰는 디렉토리 묶음."""

    root: Path
    run_id: str
    _manifest: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    # ------------------------------------------------------------------
    # 생성 / 열기
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        base_dir: Path,
        source: str,
        timestamp: datetime | None = None,
        link_latest: bool = True,
    ) -> RunWorkspace:
        """``base_dir`` 아래에 새 run 디렉토리를 만든다.

        Args:
            base_dir: ``runs/`` 에 해당하는 디렉토리.
            source: 입력 경로 또는 URL. run_id 슬러그에 쓰인다.
            timestamp: 테스트에서 결정론을 위해 주입. 기본값은 현재 시각.
            link_latest: ``latest`` 심볼릭 링크 갱신 여부.
        """
        stamp = timestamp or datetime.now()
        run_id = f"{stamp.strftime(_RUN_ID_TIME_FORMAT)}_{slugify(source)}"

        root = base_dir / run_id
        # 같은 초에 두 번 실행되면 접미사를 붙여 덮어쓰기를 막는다.
        suffix = 2
        while root.exists():
            root = base_dir / f"{run_id}-{suffix}"
            suffix += 1
        run_id = root.name

        workspace = cls(root=root, run_id=run_id)
        for directory in (workspace.input_dir, workspace.stages_dir, workspace.output_dir):
            directory.mkdir(parents=True, exist_ok=True)

        workspace._manifest = {
            "run_id": run_id,
            "created_at": stamp.isoformat(timespec="seconds"),
            "source": str(source),
            "git_commit": current_git_commit(),
            "config": {},
            "stages": [],
            "outputs": {},
            "totals": {},
        }
        workspace.save_manifest()

        if link_latest:
            workspace._update_latest_link(base_dir)
        return workspace

    @classmethod
    def open(cls, root: Path) -> RunWorkspace:
        """기존 run 디렉토리를 연다."""
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f"run 디렉토리가 없습니다: {root}")
        workspace = cls(root=root, run_id=root.name)
        if workspace.manifest_path.exists():
            workspace._manifest = json.loads(workspace.manifest_path.read_text())
        return workspace

    def _update_latest_link(self, base_dir: Path) -> None:
        link = base_dir / LATEST_LINK
        try:
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(self.root.name)
        except OSError:  # pragma: no cover - 심볼릭 링크 미지원 파일시스템
            pass

    # ------------------------------------------------------------------
    # 경로
    # ------------------------------------------------------------------

    @property
    def input_dir(self) -> Path:
        return self.root / "input"

    @property
    def stages_dir(self) -> Path:
        return self.root / "stages"

    @property
    def output_dir(self) -> Path:
        return self.root / "output"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def stage(self, name: str) -> Path:
        """단계 산출물 경로(``stages/<name>``)."""
        return self.stages_dir / name

    def output(self, name: str) -> Path:
        """최종 결과 경로(``output/<name>``)."""
        return self.output_dir / name

    def relative(self, path: Path) -> str:
        """manifest 기록용 상대 경로 문자열."""
        try:
            return str(Path(path).relative_to(self.root))
        except ValueError:
            return str(path)

    # ------------------------------------------------------------------
    # manifest
    # ------------------------------------------------------------------

    @property
    def manifest(self) -> dict[str, Any]:
        return self._manifest

    def set_config(self, **config: Any) -> None:
        self._manifest.setdefault("config", {}).update(config)

    def record_stage(
        self,
        name: str,
        count: int | None = None,
        elapsed_sec: float | None = None,
        path: Path | None = None,
        **extra: Any,
    ) -> None:
        """단계 하나의 실행 결과를 manifest 에 누적한다."""
        entry: dict[str, Any] = {"stage": name}
        if count is not None:
            entry["count"] = count
        if elapsed_sec is not None:
            entry["elapsed_sec"] = round(float(elapsed_sec), 3)
        if path is not None:
            entry["path"] = self.relative(path)
        entry.update(extra)
        self._manifest.setdefault("stages", []).append(entry)

    def record_output(self, kind: str, path: Path) -> None:
        self._manifest.setdefault("outputs", {})[kind] = self.relative(path)

    def record_totals(self, **totals: Any) -> None:
        self._manifest.setdefault("totals", {}).update(totals)

    def save_manifest(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            json.dumps(self._manifest, indent=2, ensure_ascii=False, default=str)
        )
        return self.manifest_path


def list_runs(base_dir: Path) -> list[Path]:
    """``base_dir`` 안의 run 디렉토리를 최신순으로 반환한다(latest 링크 제외)."""
    if not Path(base_dir).is_dir():
        return []
    runs = [
        p
        for p in Path(base_dir).iterdir()
        if p.is_dir() and not p.is_symlink() and p.name != LATEST_LINK
    ]
    return sorted(runs, key=lambda p: p.name, reverse=True)
