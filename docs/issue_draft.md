**Title:** Proposal: Persian (fa) diagnostic benchmark, in the style of feishu_zh

Hi, and thanks for Laya.

Following the example of `research/benchmarks/feishu_zh` (#154), I'd like to contribute a small
**Persian** diagnostic benchmark. As far as I can tell, no existing suite covers Persian.

**What it is**
- 64 synthetic Persian support messages, 4 balanced labels (`billing` / `technical` / `cancel` / `other`)
- 8 families that target Persian-specific surface forms: formal, colloquial, Finglish (Latin-script
  Persian), orthography (Arabic ي/ك, missing ZWNJ), code-mixed Persian/English, negation,
  sarcasm/taarof, and Persian vs Latin digits
- two frozen prompt modes (Persian and English instructions, identical option keys)
- a standard-library-only `audit.py` (dataset/prompt hashes, completeness, finite probabilities,
  argmax consistency, recomputed metrics) and model-free unit tests, so CI needs no weights
- highlights **false `cancel`** (wrongly closing an account) as well as accuracy, plus per-trap
  results (for example negated cancellation, which relates to #377)
- an optional normalisation ablation (ي/ك, digits, ZWNJ)

**Honesty notes:** the texts are AI-assisted synthetic fixtures reviewed by a native speaker, not
real customer data. It is a regression diagnostic, not a model ranking.

Would you be interested in a PR adding it under `research/benchmarks/persian_fa/`? I'm happy
to adapt the layout or scope to match your conventions.
