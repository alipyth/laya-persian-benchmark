"""Offline audit. Standard library only: no model, no network, no API keys.

    python audit.py                         # check the dataset and manifest only
    python audit.py --run-dir runs/x        # also check a run and recompute its metrics
    python audit.py --run-dir runs/x --write   # ...and write summary.json / summary.md into it

Exit code is non-zero when anything fails, so it can gate CI.
"""
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

from common import FAMILIES, PINNED_FILES, load_cases, load_jsonl, load_manifest, pinned_hashes
from metrics import repeat_agreement, score
from prompts import LABELS, MODES

REQUIRED_FIELDS = ("id", "family", "label", "text", "rationale", "trap")


class Audit:
    def __init__(self):
        self.problems, self.warnings = [], []

    def fail(self, msg):
        self.problems.append(msg)

    def warn(self, msg):
        self.warnings.append(msg)


# --------------------------------------------------------------------------- dataset
def check_dataset(a: Audit, cases: list):
    ids = [c.get("id") for c in cases]
    for cid, n in Counter(ids).items():
        if n > 1:
            a.fail(f"duplicate case id {cid}")
    for c in cases:
        missing = [f for f in REQUIRED_FIELDS if f not in c]
        if missing:
            a.fail(f"{c.get('id')}: missing fields {missing}")
        if c.get("label") not in LABELS:
            a.fail(f"{c.get('id')}: label {c.get('label')!r} not in {LABELS}")
        if c.get("family") not in FAMILIES:
            a.fail(f"{c.get('id')}: unknown family {c.get('family')!r}")
        if not str(c.get("text", "")).strip():
            a.fail(f"{c.get('id')}: empty text")
    # balance: every family has every label equally often
    grid = Counter((c.get("family"), c.get("label")) for c in cases)
    per_cell = {grid[(f, lab)] for f in FAMILIES for lab in LABELS}
    if len(per_cell) != 1 or 0 in per_cell:
        a.fail(f"families x labels is not balanced: {dict(grid)}")
    texts = Counter(c.get("text") for c in cases)
    for t, n in texts.items():
        if n > 1:
            a.fail(f"duplicate text: {t[:40]}...")


def check_manifest(a: Audit, cases: list):
    try:
        manifest = load_manifest()
    except FileNotFoundError:
        a.fail("data/manifest.json missing; run python tools/update_manifest.py")
        return
    current = pinned_hashes()
    for rel in PINNED_FILES:
        if manifest.get("sha256", {}).get(rel) != current[rel]:
            a.fail(f"{rel} changed since the manifest was written (hash mismatch)")
    if manifest.get("n_cases") != len(cases):
        a.fail(f"manifest n_cases={manifest.get('n_cases')} but dataset has {len(cases)}")
    if manifest.get("reviewed_by", "TODO") == "TODO":
        a.warn("manifest reviewed_by is TODO: have a native speaker review every text and label before publishing")


# --------------------------------------------------------------------------- run
def check_run(a: Audit, cases: list, run_dir: Path):
    meta_path, resp_path = run_dir / "metadata.json", run_dir / "responses.jsonl"
    if not meta_path.exists() or not resp_path.exists():
        a.fail(f"{run_dir}: needs metadata.json and responses.jsonl")
        return None, None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    rows = load_jsonl(resp_path)

    if meta.get("sha256") != pinned_hashes():
        a.fail("run was recorded against different data/prompts (sha256 mismatch)")
    if meta.get("smoke_test"):
        a.warn("smoke-test run (--limit): not a full-protocol result")
    if meta.get("not_a_model_result"):
        a.warn("keyword matcher: pipeline check only, not a model result or baseline")

    case_ids = meta.get("case_ids") or [c["id"] for c in cases]
    known = {c["id"] for c in cases}
    for cid in case_ids:
        if cid not in known:
            a.fail(f"run lists unknown case {cid}")

    expected = {(m, r, cid) for m in meta["modes"] for r in range(meta["repeats"]) for cid in case_ids}
    seen = Counter((row["mode"], row["repeat"], row["case_id"]) for row in rows)
    for key, n in seen.items():
        if n > 1:
            a.fail(f"duplicate response {key}")
        if key not in expected:
            a.fail(f"unexpected response {key}")
    for key in expected - set(seen):
        a.fail(f"missing response {key}")

    for row in rows:
        tag = f"{row['case_id']}/{row['mode']}/r{row['repeat']}"
        if row.get("error"):
            continue  # counted as wrong by metrics, not an audit failure
        pred, probs = row.get("prediction"), row.get("probabilities")
        if pred not in LABELS:
            a.fail(f"{tag}: prediction {pred!r} not a label")
            continue
        if not isinstance(probs, dict) or set(probs) != set(LABELS):
            a.fail(f"{tag}: probabilities must cover exactly {LABELS}")
            continue
        vals = list(probs.values())
        if not all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1 for v in vals):
            a.fail(f"{tag}: non-finite or out-of-range probability")
        elif abs(sum(vals) - 1) > 0.02:
            a.fail(f"{tag}: probabilities sum to {sum(vals):.4f}")
        elif probs[pred] < max(vals) - 1e-6:
            a.fail(f"{tag}: prediction is not the argmax")
    return meta, rows


# --------------------------------------------------------------------------- report
def pct(x):
    return "-" if x is None else f"{100 * x:.1f}%"


def summarise(cases, meta, rows):
    run_cases = [c for c in cases if c["id"] in set(meta.get("case_ids") or [c["id"] for c in cases])]
    out = {"backend": meta.get("backend"), "normalize": meta.get("normalize"), "modes": {}}
    for mode in meta["modes"]:
        s = score(run_cases, rows, mode, repeat=0)
        s["repeat_agreement"] = repeat_agreement(rows, mode)
        out["modes"][mode] = s
    return out


def render_markdown(summary):
    b = summary["backend"] or {}
    title = b.get("checkpoint") or b.get("backend")
    lines = [f"# Results: {title}{' + normalize' if summary.get('normalize') else ''}", ""]
    for mode, s in summary["modes"].items():
        ece = "-" if s["ece"] is None else "%.3f" % s["ece"]
        p50 = "-" if s["latency_ms_p50"] is None else "%.1f ms" % s["latency_ms_p50"]
        lines += [f"## {mode}", "",
                  f"- accuracy **{s['correct']}/{s['n']} ({pct(s['accuracy'])})**, macro-F1 {s['macro_f1']:.3f}, failures {s['failures']}",
                  f"- false `cancel` (would wrongly close an account): **{s['false_cancel']}**; missed `cancel`: {s['missed_cancel']}",
                  f"- mean answer confidence {pct(s['mean_answer_confidence'])}, ECE {ece}",
                  f"- latency p50 {p50}, repeat agreement {pct(s['repeat_agreement'])}",
                  "", "| family | correct |", "|---|---|"]
        for fam, v in s["by_family"].items():
            lines.append(f"| {fam} | {v['correct']}/{v['n']} |")
        lines += ["", "| gold \\ predicted | " + " | ".join(LABELS) + " | failed |",
                  "|---" * (len(LABELS) + 2) + "|"]
        for g in LABELS:
            row = s["confusion"][g]
            lines.append(f"| {g} | " + " | ".join(str(row.get(p, 0)) for p in LABELS) + f" | {row.get('FAILED', 0)} |")
        lines.append("")
    lines.append("Quality is scored on the first repeat (preselected, not the best). Failures count as wrong.")
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--write", action="store_true", help="write summary.json and summary.md into the run dir")
    args = p.parse_args(argv)

    a = Audit()
    cases = load_cases()
    check_dataset(a, cases)
    check_manifest(a, cases)
    print(f"dataset: {len(cases)} cases, {len(FAMILIES)} families, {len(LABELS)} labels, {len(MODES)} prompt modes")

    summary = None
    if args.run_dir:
        meta, rows = check_run(a, cases, args.run_dir)
        if meta is not None and not a.problems:
            summary = summarise(cases, meta, rows)
            print()
            print(render_markdown(summary))
            if args.write:
                (args.run_dir / "summary.json").write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                (args.run_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")
                print(f"wrote {args.run_dir}/summary.json and summary.md")

    for w in a.warnings:
        print(f"WARNING: {w}")
    if a.problems:
        for prob in a.problems[:50]:
            print(f"FAIL: {prob}")
        print(f"\naudit FAILED ({len(a.problems)} problems)")
        return 1
    print("audit OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
