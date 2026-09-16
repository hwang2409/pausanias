from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .bench import format_report, run_benchmark
from .config import ConfigError, default_config_path, load_config
from .core import excerpt, index, read_source, search
from .model_bundle import (
    BundleError,
    MODEL_BUNDLE_MANIFEST,
    NUMPY_VERSION,
    ONNXRUNTIME_VERSION,
    fetch_bundle,
    license_records,
    package_version,
    verify_bundle,
)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="pausanias")
    root.add_argument("--config", default=str(default_config_path()))
    commands = root.add_subparsers(dest="command", required=True)
    index_parser = commands.add_parser("index")
    index_parser.add_argument("--config", dest="config", default=argparse.SUPPRESS)
    index_parser.add_argument("--rebuild", action="store_true")
    search_parser = commands.add_parser("search")
    search_parser.add_argument("--config", dest="config", default=argparse.SUPPRESS)
    search_parser.add_argument("query")
    search_parser.add_argument("--project")
    search_parser.add_argument("--root")
    search_parser.add_argument("--all-projects", action="store_true")
    search_parser.add_argument("--limit", type=int, default=20)
    search_parser.add_argument("--json", action="store_true")
    read_parser = commands.add_parser("read")
    read_parser.add_argument("--config", dest="config", default=argparse.SUPPRESS)
    read_parser.add_argument("path")
    read_parser.add_argument("--heading")
    read_parser.add_argument("--max-bytes", type=int, default=20000)
    bench_parser = commands.add_parser("bench", help="measure index and search performance")
    bench_parser.add_argument("--files", "--file-count", dest="file_count", type=int, default=2000,
                              help="synthetic Markdown files to generate (default: 2000)")
    bench_parser.add_argument("--queries", dest="query_count", type=int, default=200,
                              help="warm search queries to measure (default: 200)")
    bench_parser.add_argument("--seed", type=int, default=0, help="synthetic corpus seed (default: 0)")
    bench_parser.add_argument("--corpus-dir", help="reuse or create a synthetic corpus at DIR")
    bench_parser.add_argument(
        "--real",
        metavar="DIR",
        help="benchmark an existing Markdown directory read-only; results reflect local cost only",
    )
    bench_parser.add_argument("--json", action="store_true", help="write the report as JSON")
    model_parser = commands.add_parser("model", help="manage the optional semantic model")
    model_commands = model_parser.add_subparsers(dest="model_command", required=True)
    fetch_parser = model_commands.add_parser("fetch", help="download and verify the semantic model")
    fetch_parser.add_argument("--config", dest="config", default=argparse.SUPPRESS)
    fetch_parser.add_argument("--bundle-dir")
    license_parser = model_commands.add_parser("license", help="show model and runtime license records")
    license_parser.add_argument("--config", dest="config", default=argparse.SUPPRESS)
    license_parser.add_argument("--bundle-dir")
    status_parser = model_commands.add_parser("status", help="show semantic model readiness")
    status_parser.add_argument("--config", dest="config", default=argparse.SUPPRESS)
    status_parser.add_argument("--bundle-dir")
    status_parser.add_argument("--json", action="store_true")
    return root


def _result(candidate) -> dict:
    return {
        "excerpt": excerpt(candidate.text),
        "path": candidate.canonical_path,
        "heading": list(candidate.heading_path),
        "line_range": [candidate.line_start, candidate.line_end],
        "dates": {"updated": candidate.updated_date, "created": candidate.created_date},
        "score": candidate.score,
        "reason": candidate.reason,
        "content_hash": candidate.content_hash,
    }


def _model_status(bundle_dir: str | Path) -> dict:
    path = Path(bundle_dir).expanduser()
    runtime_version = package_version("onnxruntime")
    numpy_version = package_version("numpy")
    bundle_present = path.is_file() and not path.is_symlink()
    manifest_valid = False
    manifest_error = None
    if bundle_present:
        try:
            verify_bundle(path)
        except BundleError as exc:
            manifest_error = str(exc)
        else:
            manifest_valid = True
    runtime = MODEL_BUNDLE_MANIFEST["runtime"]
    expected_runtime = runtime["version"] if isinstance(runtime, dict) else ONNXRUNTIME_VERSION
    return {
        "bundle_dir": str(path),
        "extra_installed": runtime_version is not None and numpy_version is not None,
        "bundle_present": bundle_present,
        "manifest_valid": manifest_valid,
        "manifest_error": manifest_error,
        "versions": {
            "onnxruntime": {"installed": runtime_version, "expected": expected_runtime},
            "numpy": {
                "installed": numpy_version,
                "expected": MODEL_BUNDLE_MANIFEST.get("numpy_version", NUMPY_VERSION),
            },
        },
    }


def _configured_bundle(args) -> Path:
    if args.bundle_dir is not None:
        return Path(args.bundle_dir)
    return load_config(args.config).bundle_dir


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "bench":
            report = run_benchmark(
                file_count=args.file_count,
                query_count=args.query_count,
                seed=args.seed,
                corpus_dir=args.corpus_dir,
                real_dir=args.real,
            )
            if args.json:
                print(json.dumps(report, sort_keys=True))
            else:
                print(format_report(report))
            return 1 if report["gate"]["enforced"] and not report["gate"]["passed"] else 0
        if args.command == "model" and args.model_command == "license":
            bundle_dir = args.bundle_dir
            if bundle_dir is None:
                try:
                    bundle_dir = _configured_bundle(args)
                except ConfigError:
                    if Path(args.config).expanduser().exists():
                        raise
                    bundle_dir = None
            for record in license_records(bundle_dir):
                print(f"{record.get('name', 'license')}: {record.get('spdx_id', 'unknown')}")
                if isinstance(record.get("notice"), str):
                    print(record["notice"], end="" if record["notice"].endswith("\n") else "\n")
            return 0
        if args.command == "model" and args.model_command == "status":
            status = _model_status(_configured_bundle(args))
            if args.json:
                print(json.dumps(status, sort_keys=True))
            else:
                print(f"bundle directory: {status['bundle_dir']}")
                print(f"extra installed: {'yes' if status['extra_installed'] else 'no'}")
                print(f"bundle present: {'yes' if status['bundle_present'] else 'no'}")
                print(f"manifest valid: {'yes' if status['manifest_valid'] else 'no'}")
                for name, versions in status["versions"].items():
                    print(f"{name}: installed={versions['installed'] or 'missing'} expected={versions['expected']}")
                if status["manifest_error"]:
                    print(f"manifest error: {status['manifest_error']}")
            return 0
        if args.command == "model" and args.model_command == "fetch":
            bundle_dir = _configured_bundle(args)
            print(f"model bundle installed at {fetch_bundle(Path(bundle_dir))}")
        elif args.command == "index":
            config = load_config(args.config)
            print(f"index generation {index(config, args.rebuild)}")
        elif args.command == "read":
            config = load_config(args.config)
            print(read_source(config, args.path, args.heading, args.max_bytes))
        else:
            config = load_config(args.config)
            results = [_result(item) for item in search(config, args.query, args.project, args.root, args.all_projects, args.limit)]
            if args.json:
                print(json.dumps(results, ensure_ascii=False))
            else:
                for item in results:
                    print(f"{item['path']}:{item['line_range'][0]}-{item['line_range'][1]} {item['heading']}")
                    print(f"  {item['excerpt']}")
                    print(f"  reason: {item['reason']}")
        return 0
    except (ConfigError, ValueError, OSError) as exc:
        print(f"pausanias: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
