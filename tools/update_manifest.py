"""Rewrite data/manifest.json after an intentional change to the data or prompts.

    python tools/update_manifest.py

Only do this before publishing results, or together with a new results version: any run
recorded against the old hashes will (correctly) fail the audit afterwards.
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import MANIFEST_PATH, load_cases, pinned_hashes  # noqa: E402
from prompts import LABELS, MODES  # noqa: E402


def build() -> dict:
    cases = load_cases()
    return {
        "name": "persian_fa",
        "title": "Persian support-triage decisions",
        "version": "1.0",
        "task": "4-way choice: which department handles a Persian support message",
        "labels": list(LABELS),
        "modes": list(MODES),
        "n_cases": len(cases),
        "families": dict(Counter(c["family"] for c in cases)),
        "label_counts": dict(Counter(c["label"] for c in cases)),
        "sha256": pinned_hashes(),
        "label_policy": (
            "Label = the department that must act on the message's main request. "
            "A mentioned-but-negated topic ('I don't want to cancel, just ...') does not set the label. "
            "Sarcastic thanks is labelled by the complaint it carries; genuine thanks is 'other'. "
            "Stopping auto-renewal or closing/deleting the account is 'cancel'."
        ),
        "provenance": (
            "Synthetic messages written for this benchmark, drafted with AI assistance. "
            "No real customer data."
        ),
        # Fill in before publishing: who reviewed every text and label, and how.
        "reviewed_by": previous_reviewer(),
    }


def previous_reviewer() -> str:
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8")).get("reviewed_by", "TODO")
    except FileNotFoundError:
        return "TODO"


if __name__ == "__main__":
    MANIFEST_PATH.write_text(json.dumps(build(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {MANIFEST_PATH}")
