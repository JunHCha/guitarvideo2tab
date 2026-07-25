"""CLI entry point: `python -m guitarvideo2tab <input> [options]`."""
from __future__ import annotations

# --- Silence third-party startup noise (must run before heavy imports) -------
import logging
import os
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("ABSL_LOGGING_VERBOSITY", "3")
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)
# -----------------------------------------------------------------------------

# E402: 아래 import 들은 위의 로그 억제 설정이 먼저 실행되어야 효과가 있으므로
# 의도적으로 모듈 상단이 아닌 이 위치에 둔다.
import argparse  # noqa: E402
import contextlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

# coremltools / basic-pitch emit cosmetic print()s at import time; mute stdout
# while heavy deps load (stderr stays live so real import errors still surface).
with contextlib.redirect_stdout(io.StringIO()):
    from .pipeline import Pipeline  # noqa: E402
    from .workspace import RUNS_DIRNAME, RunWorkspace, list_runs  # noqa: E402


def _cmd_run(args: argparse.Namespace) -> int:
    workspace = RunWorkspace.create(args.runs_dir, args.input)
    print(f"▶ run: {workspace.run_id}")
    print(f"  workspace: {workspace.root}")

    pipeline = Pipeline(
        workspace=workspace,
        save_intermediates=not args.no_intermediates,
        tempo_bpm=args.tempo,
        subdivision=args.subdivision,
    )
    outputs = pipeline.run(args.input)

    totals = workspace.manifest.get("totals", {})
    print("\n✓ 완료")
    for kind, path in outputs.items():
        print(f"  {kind:<9} {path}")
    print(f"  manifest  {workspace.manifest_path}")
    if totals:
        print(
            f"\n  tempo {totals.get('tempo_bpm')}bpm · {totals.get('measures')}마디 · "
            f"음표 {totals.get('notes_written')}개 (화음 {totals.get('chords')}) · "
            f"운지 일치율 {totals.get('fingering_match_ratio')}"
        )
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    runs = list_runs(args.runs_dir)
    if not runs:
        print(f"실행 기록이 없습니다: {args.runs_dir}")
        return 0
    for run in runs:
        manifest_path = run / "manifest.json"
        summary = ""
        if manifest_path.exists():
            try:
                totals = json.loads(manifest_path.read_text()).get("totals", {})
                summary = (
                    f"  {totals.get('measures', '?')}마디"
                    f" 음표 {totals.get('notes_written', '?')}"
                    f" {totals.get('elapsed_sec', '?')}s"
                )
            except (OSError, ValueError):
                summary = "  (manifest 읽기 실패)"
        print(f"{run.name}{summary}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="guitarvideo2tab")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path(RUNS_DIRNAME),
        help="실행별 산출물이 쌓이는 디렉토리 (기본: runs/)",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="파이프라인 실행 (기본 명령)")
    run_parser.add_argument("input", help="YouTube URL 또는 로컬 영상 파일 경로")
    run_parser.add_argument("--tempo", type=float, default=None, help="BPM 고정 (미지정 시 추정)")
    run_parser.add_argument(
        "--subdivision", type=int, default=16, help="양자화 해상도 (기본 16분음표)"
    )
    run_parser.add_argument(
        "--no-intermediates", action="store_true", help="단계별 JSON dump 생략"
    )
    run_parser.set_defaults(func=_cmd_run)

    list_parser = subparsers.add_parser("list", help="지난 실행 목록")
    list_parser.set_defaults(func=_cmd_list)

    # 하위 명령 없이 `python -m guitarvideo2tab <input>` 로도 실행되게 한다.
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in {"run", "list", "-h", "--help"} and not argv[0].startswith("-"):
        argv = ["run", *argv]

    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
