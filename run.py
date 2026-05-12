"""
run.py — Egyptian Road & Mobility Assistant
============================================
Master CLI for all pipeline steps.

Usage:
    python run.py scrape               # scrape all domains
    python run.py scrape --group driving_license   # one domain only
    python run.py index                # build FAISS index
    python run.py serve                # start API (localhost:8000)
    python run.py eval                 # run full evaluation
    python run.py eval --domain traffic_law
    python run.py eval --no-hyde       # measure HyDE's contribution
    python run.py eval --output eval/results.json
    python run.py all                  # scrape + index + serve
"""

import sys
import subprocess
import argparse
from pathlib import Path

ROOT = Path(__file__).parent


def run_scrape(group: str | None = None) -> int:
    cmd = [sys.executable, "-m", "scraper.scraper"]
    if group:
        cmd += ["--group", group]
    return subprocess.run(cmd, cwd=ROOT).returncode


def run_index() -> int:
    return subprocess.run(
        [sys.executable, "-m", "indexer.faiss_store"], cwd=ROOT
    ).returncode


def run_serve() -> int:
    return subprocess.run(
        [
            sys.executable, "-m", "uvicorn",
            "api.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload",
        ],
        cwd=ROOT,
    ).returncode


def run_eval(domain: str | None, output: str | None, no_hyde: bool, no_rerank: bool) -> int:
    cmd = [sys.executable, "-m", "eval.evaluate"]
    if domain:
        cmd += ["--domain", domain]
    if output:
        cmd += ["--output", output]
    if no_hyde:
        cmd += ["--no-hyde"]
    if no_rerank:
        cmd += ["--no-rerank"]
    return subprocess.run(cmd, cwd=ROOT).returncode


def main():
    parser = argparse.ArgumentParser(
        description="Egyptian Road & Mobility Assistant — pipeline CLI"
    )
    subparsers = parser.add_subparsers(dest="command")

    # scrape
    p_scrape = subparsers.add_parser("scrape", help="Scrape web sources")
    p_scrape.add_argument(
        "--group", type=str, default=None,
        help=(
            "Scrape only this domain group. "
            "Options: traffic_law, driving_license, vehicle_registration, "
            "accident_liability, commercial_vehicles, driver_fitness, "
            "international_driving, road_infrastructure"
        ),
    )

    # index
    subparsers.add_parser("index", help="Build FAISS index from scraped chunks")

    # serve
    subparsers.add_parser("serve", help="Start the FastAPI server")

    # eval
    p_eval = subparsers.add_parser("eval", help="Run evaluation suite")
    p_eval.add_argument("--domain",   type=str,  default=None, help="Evaluate one domain only")
    p_eval.add_argument("--output",   type=str,  default=None, help="Save report to JSON file")
    p_eval.add_argument("--no-hyde",  action="store_true",     help="Disable HyDE")
    p_eval.add_argument("--no-rerank",action="store_true",     help="Disable reranker")

    # all
    subparsers.add_parser("all", help="scrape + index + serve")

    args = parser.parse_args()

    if args.command == "scrape":
        sys.exit(run_scrape(group=getattr(args, "group", None)))
    elif args.command == "index":
        sys.exit(run_index())
    elif args.command == "serve":
        sys.exit(run_serve())
    elif args.command == "eval":
        sys.exit(run_eval(
            domain=args.domain,
            output=args.output,
            no_hyde=args.no_hyde,
            no_rerank=args.no_rerank,
        ))
    elif args.command == "all":
        for fn in [run_scrape, run_index, run_serve]:
            code = fn() if fn != run_scrape else run_scrape()
            if code != 0 and fn != run_serve:
                print(f"Step failed with exit code {code}")
                sys.exit(code)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
