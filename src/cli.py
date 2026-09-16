from __future__ import annotations

import argparse
import json
import sys

from .bench import format_report, run_benchmark
from .config import ConfigError, default_config_path, load_config
from .core import excerpt, index, read_source, search


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
        config = load_config(args.config)
        if args.command == "index":
            print(f"index generation {index(config, args.rebuild)}")
        elif args.command == "read":
            print(read_source(config, args.path, args.heading, args.max_bytes))
        else:
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
