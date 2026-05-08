"""CLI entry point: `python -m guitarvideo2tab <input> [--output OUT]`."""
from __future__ import annotations

import argparse
from pathlib import Path

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
