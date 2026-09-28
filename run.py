"""Run the benchmark against one backend and record every raw response.

Examples (from this folder)::

    # pipeline check, no model download (keyword matcher: NOT a model result or baseline)
    python run.py --backend keyword --output runs/keyword

    # smoke test on the real model: 2 cases, 1 repeat, 1 mode
    python run.py --backend laya --checkpoint multilingual --limit 2 --repeats 1 \
        --modes choice_fa --output runs/smoke

    # full protocol (64 cases x 2 modes x 3 repeats)
    python run.py --backend laya --checkpoint multilingual --output runs/multilingual

Every run goes to a NEW directory containing ``metadata.json`` and ``responses.jsonl``.
Check it with ``python audit.py --run-dir <dir>``.
"""
import argparse
import datetime as dt
import json
import platform
import sys
import time
import traceback
from pathlib import Path

from common import load_cases, pinned_hashes
from normalize import normalize
from prompts import LABELS, MODES, QUESTION_ID, build_state

NAMED_CHECKPOINTS = {"english": None, "multilingual": "multilingual", "typed-decisions": "typed-decisions"}
HUB_REPO = "convaiinnovations/laya"


# --------------------------------------------------------------------------- backends
class KeywordBackend:
    """Deterministic keyword matcher, used only to test the pipeline without downloading
    weights. It is NOT a model result and NOT a fair baseline: its keywords were written with
    these 64 messages in view, so its score is inflated. Never publish it as a comparison."""

    name = "keyword"
    KEYWORDS = {
        "cancel": ("لغو", "کنسل", "ببندید", "حذف", "غیرفعال", "غيرفعال", "cancel", "deactivate",
                   "delete", "laghv"),
        "billing": ("پول", "مبلغ", "فاکتور", "فاكتور", "قیمت", "پرداخت", "کسر", "كسر", "هزینه",
                    "invoice", "refund", "charge", "pardakht", "pool", "factor", "شارژ"),
        "technical": ("خطا", "ارور", "باگ", "کرش", "باز نمی", "نمیشه", "نمی‌شه", "لاگ", "login",
                      "رمز", "کد تأیید", "api", "کند", "قطع", "baz nemishe", "error"),
    }

    def describe(self):
        return {"backend": "keyword"}

    def predict(self, state, questions):
        text = state["message"].lower()
        scores = {lab: sum(kw.lower() in text for kw in kws) for lab, kws in self.KEYWORDS.items()}
        best = max(scores, key=scores.get)
        pred = best if scores[best] > 0 else "other"
        probs = {lab: (0.7 if lab == pred else 0.1) for lab in LABELS}
        return {"answers": {QUESTION_ID: {"type": "choice", "choice": pred,
                                          "probabilities": probs, "answer_confidence": 0.7}}}


class LayaBackend:
    name = "laya"

    def __init__(self, checkpoint, device, revision):
        import laya  # imported lazily so the audit and tests never need torch

        self._laya = laya
        self.checkpoint = checkpoint
        self.revision = revision
        if checkpoint in NAMED_CHECKPOINTS:
            self.source = HUB_REPO
            self.subfolder = NAMED_CHECKPOINTS[checkpoint]
        else:  # a local directory or another hub repo id
            self.source, self.subfolder = checkpoint, None
        # Older laya releases have no `revision` (and some no `subfolder`) keyword, so pass
        # only what the installed version accepts.
        import inspect
        accepted = inspect.signature(laya.load).parameters
        kwargs = {"device": device}
        if self.subfolder is not None:
            if "subfolder" not in accepted:
                sys.exit(f"installed laya {getattr(laya, '__version__', '?')} cannot load subfolder "
                         f"checkpoints; upgrade with:  python -m pip install -U laya")
            kwargs["subfolder"] = self.subfolder
        if revision is not None:
            if "revision" not in accepted:
                sys.exit(f"installed laya {getattr(laya, '__version__', '?')} does not support --revision; "
                         f"upgrade with:  python -m pip install -U laya   (or drop --revision)")
            kwargs["revision"] = revision
        self.agent = laya.load(self.source, **kwargs)

    def describe(self):
        info = {"backend": "laya", "checkpoint": self.checkpoint, "source": self.source,
                "subfolder": self.subfolder, "revision_requested": self.revision,
                "laya_version": getattr(self._laya, "__version__", None)}
        for attr in ("device", "dtype"):
            if hasattr(self.agent, attr):
                info[attr] = str(getattr(self.agent, attr))
        return info

    def predict(self, state, questions):
        return self.agent.predict(state, questions)


# --------------------------------------------------------------------------- helpers
def environment():
    env = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("torch", "transformers", "numpy"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:  # not installed is fine for the keyword backend
            env[mod] = None
    return env


def to_row(case, mode, repeat, result, latency_ms, error):
    row = {"case_id": case["id"], "mode": mode, "repeat": repeat, "prediction": None,
           "probabilities": None, "answer_confidence": None,
           "latency_ms": round(latency_ms, 3) if latency_ms is not None else None, "error": error}
    if result is not None:
        ans = result["answers"][QUESTION_ID]
        row["prediction"] = ans.get("choice")
        row["probabilities"] = ans.get("probabilities")
        row["answer_confidence"] = ans.get("answer_confidence")
    return row


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=("laya", "keyword"), default="laya")
    p.add_argument("--checkpoint", default="multilingual",
                   help="english | multilingual | typed-decisions | local dir | hub repo id")
    p.add_argument("--revision", default=None, help="hub commit to pin (recommended for published runs)")
    p.add_argument("--device", default=None, help="cpu | cuda | mps (default: auto)")
    p.add_argument("--modes", nargs="+", choices=tuple(MODES), default=list(MODES))
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--limit", type=int, default=None, help="first N cases only (smoke tests)")
    p.add_argument("--normalize", action="store_true", help="ablation: Persian normalisation first")
    p.add_argument("--output", required=True, help="new directory for this run")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"refusing to overwrite non-empty {out}; use a new --output directory")

    cases = load_cases()
    if args.limit:
        cases = cases[: args.limit]

    t0 = time.perf_counter()
    backend = KeywordBackend() if args.backend == "keyword" else LayaBackend(args.checkpoint, args.device, args.revision)
    load_s = time.perf_counter() - t0
    out.mkdir(parents=True, exist_ok=True)  # only after the model loaded

    metadata = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "backend": backend.describe(),
        "environment": environment(),
        "sha256": pinned_hashes(),
        "modes": args.modes,
        "repeats": args.repeats,
        "n_cases": len(cases),
        "case_ids": [c["id"] for c in cases],
        "normalize": args.normalize,
        "smoke_test": bool(args.limit),
        "load_seconds": round(load_s, 2),
        "warmup": "one excluded warm-up request per mode",
        "not_a_model_result": args.backend == "keyword",
    }

    total = len(args.modes) * args.repeats * len(cases)
    done = errors = 0
    with open(out / "responses.jsonl", "w", encoding="utf-8") as fh:
        for mode in args.modes:
            questions = MODES[mode]
            try:  # warm-up, excluded from timing and results
                backend.predict(build_state(cases[0]["text"]), questions)
            except Exception:
                pass
            for repeat in range(args.repeats):
                for case in cases:
                    text = normalize(case["text"]) if args.normalize else case["text"]
                    result = error = None
                    start = time.perf_counter()
                    try:
                        result = backend.predict(build_state(text), questions)
                    except Exception as exc:  # recorded, never retried
                        error = f"{type(exc).__name__}"
                        errors += 1
                        if errors == 1:
                            traceback.print_exc()
                    latency = (time.perf_counter() - start) * 1000
                    fh.write(json.dumps(to_row(case, mode, repeat, result, latency, error),
                                        ensure_ascii=False) + "\n")
                    done += 1
                    print(f"\r{done}/{total}  {mode} r{repeat}  errors={errors}", end="", flush=True)
    print()
    metadata["errors"] = errors
    (out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}/responses.jsonl and metadata.json")
    print(f"next: python audit.py --run-dir {out}")


if __name__ == "__main__":
    main()
