"""Shared paths, hashing and dataset loading. Standard library only."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CASES_PATH = ROOT / "data" / "cases.jsonl"
MANIFEST_PATH = ROOT / "data" / "manifest.json"

# Files whose bytes define the benchmark. A run records their hashes; the audit refuses a
# run whose hashes do not match the current checkout.
PINNED_FILES = ("data/cases.jsonl", "prompts.py", "normalize.py")

FAMILIES = ("formal", "colloquial", "finglish", "orthography",
            "code_mixed", "negation", "sarcasm_taarof", "digits")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pinned_hashes() -> dict:
    return {rel: sha256_file(ROOT / rel) for rel in PINNED_FILES}


def load_cases(path: Path = CASES_PATH) -> list:
    cases = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name}:{i}: invalid JSON ({exc})") from None
    return cases


def load_jsonl(path: Path) -> list:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
