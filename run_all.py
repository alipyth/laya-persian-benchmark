"""One command for the whole study: two checkpoints x (raw, normalised), then audit and compare.

    python run_all.py                      # CPU, 3 repeats (roughly 20-60 min on a laptop CPU)
    python run_all.py --device cuda
    python run_all.py --repeats 1 --quick  # first 8 cases only, to check your setup

Output goes to runs/<timestamp>/.
"""
import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

CONFIGS = [
    ("multilingual", False),
    ("multilingual", True),
    ("english", False),
]


def sh(*args):
    print("\n$ " + " ".join(str(x) for x in args), flush=True)
    subprocess.run([sys.executable, *map(str, args)], cwd=HERE, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default=None)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--revision", default=None)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    root = HERE / "runs" / dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    dirs = []
    for ckpt, norm in CONFIGS:
        d = root / (ckpt + ("_norm" if norm else ""))
        cmd = ["run.py", "--backend", "laya", "--checkpoint", ckpt, "--repeats", args.repeats, "--output", d]
        if norm:
            cmd.append("--normalize")
        if args.device:
            cmd += ["--device", args.device]
        if args.revision:
            cmd += ["--revision", args.revision]
        if args.quick:
            cmd += ["--limit", 8]
        sh(*cmd)
        sh("audit.py", "--run-dir", d, "--write")
        dirs.append(d)

    for mode in ("choice_fa", "choice_en"):
        sh("compare.py", *dirs, "--mode", mode, "--out", root / f"comparison_{mode}.md")
    print(f"\nall done: {root}")


if __name__ == "__main__":
    main()
