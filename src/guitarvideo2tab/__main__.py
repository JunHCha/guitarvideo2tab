"""CLI entry point: `python -m guitarvideo2tab <input> [--output OUT]`."""
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
from pathlib import Path  # noqa: E402

# coremltools / basic-pitch emit cosmetic print()s at import time; mute stdout
# while heavy deps load (stderr stays live so real import errors still surface).
with contextlib.redirect_stdout(io.StringIO()):
    from .pipeline import Pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="guitarvideo2tab")
    parser.add_argument("input", help="YouTube URL or local video file path")
    parser.add_argument("-o", "--output", type=Path, default=Path("output.gpx"))
    parser.add_argument("--save-intermediates", action="store_true")
    parser.add_argument("--workdir", type=Path, default=Path(".cache/guitarvideo2tab"))
    args = parser.parse_args(argv)

    pipeline = Pipeline(workdir=args.workdir, save_intermediates=args.save_intermediates)
    pipeline.run(args.input, output_path=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
