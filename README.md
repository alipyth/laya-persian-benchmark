# Persian support-triage decisions (`persian_fa`)

[فارسی](README.fa.md)

A small, self-contained diagnostic for Laya on **Persian**: **64 synthetic support messages,
8 families, 4 balanced labels**, frozen prompts in two instruction languages, an offline audit,
and model-free tests.

Each message is routed to one department: `billing`, `technical`, `cancel`, or `other`. The
families target surface forms that Persian users actually write and that no existing Laya
benchmark covers:

| family | what it tests | example |
|---|---|---|
| `formal` | written/administrative register | «خواهشمند است نسبت به استرداد وجه اقدام فرمایید» |
| `colloquial` | spoken register | «پولم کم شده ولی اشتراکم فعال نشده» |
| `finglish` | Persian in Latin script | `mikham eshterakam ro laghv konam` |
| `orthography` | Arabic ي/ك, missing ZWNJ, spaced suffixes | «ميخواستم بدونم چرا مبلغ اشتراكم دوبار كسر شده» |
| `code_mixed` | Persian with English tech/billing words | «لطفاً subscription من رو cancel کنید» |
| `negation` | a topic mentioned only to be negated | «نمی‌خوام اشتراکم رو لغو کنم، فقط بگید چرا دو بار پول کم شد» |
| `sarcasm_taarof` | sarcastic thanks and polite *taarof* | «دستتون درد نکنه، سه ماهه پول دو تا اشتراک رو می‌دم!» |
| `digits` | Persian vs Latin digits, dates, amounts | «مبلغ ۱۲۹,۰۰۰ تومان در تاریخ ۱۴۰۵/۰۷/۰۲ دو بار کسر شده» |

Each family has exactly two messages per label. Many cases carry a `trap` tag
(`negated_cancel`, `sarcastic_thanks`, `billing_word_not_billing`, ...) so results can be
read per failure mechanism, not just per family.

**The messages and reference labels are synthetic fixtures drafted with AI assistance, not
real customer data.** This is a diagnostic, not a general model ranking or an independently
annotated blind test. See [Limitations](#limitations).

## Start here: verify offline

Python 3.10+, standard library only: no model, no network, no API key.

```bash
python audit.py
python -m unittest discover -s tests -v
```

The audit checks unique IDs, the balanced family x label grid, label validity, and the
SHA-256 of `data/cases.jsonl`, `prompts.py` and `normalize.py` against `data/manifest.json`.

## Run Laya

```bash
pip install -r requirements.txt

# smoke test: 2 cases, 1 repeat, 1 mode
python run.py --backend laya --checkpoint multilingual --limit 2 --repeats 1 \
    --modes choice_fa --output runs/smoke
python audit.py --run-dir runs/smoke

# full protocol: 64 cases x 2 prompt modes x 3 repeats
python run.py --backend laya --checkpoint multilingual --output runs/multilingual
python audit.py --run-dir runs/multilingual --write
```

Or run the whole study (multilingual, multilingual + normalisation, english), then audit and
compare, with one command:

```bash
python run_all.py                 # add --device cuda|mps, --revision <hub commit>
```

- `--checkpoint`: `multilingual`, `english`, `typed-decisions`, a local checkpoint directory,
  or a Hub repo id (for example your own fine-tune).
- `--revision`: pin the Hub commit. Do this for any run you publish.
- `--normalize`: ablation. Maps Arabic ي/ك to Persian ی/ک, digits to ASCII, and restores ZWNJ
  in common spots (`normalize.py`). Finglish is left as is.
- Every run writes to a **new** directory: `metadata.json` (backend, versions, device, data and
  prompt hashes) and `responses.jsonl` (every raw answer, probabilities and latency). One
  warm-up request per mode is excluded. Requests are not retried.

`--backend keyword` runs a keyword matcher so the pipeline can be tested without weights. Its
keywords were written with these messages in view, so **its score is not a baseline** and must
not be published as one.

## Metrics

`audit.py --write` recomputes everything from the raw responses and writes `summary.md` and
`summary.json`:

- accuracy and macro-F1, **scored on the first repeat** (preselected, not the best);
- per family, per label, per trap, and a confusion matrix;
- **false `cancel`**: messages that would wrongly close someone's account (the costliest
  error in support triage) and missed `cancel`;
- ECE and mean confidence over `answer_confidence` (the calibrated probability of the answer);
- p50/p90 latency and repeat agreement.

Failed requests stay in the denominator and count as wrong.

`compare.py` puts several audited runs side by side, per family:

```bash
python compare.py runs/multilingual runs/english runs/multilingual_norm --mode choice_fa
```

## Prompt modes

The same four options are asked two ways (`prompts.py`):

- `choice_fa`: instructions and option descriptions in Persian;
- `choice_en`: the same content in English.

Option keys are identical and deliberately non-boolean, following the README's warning about
`yes`/`no`-style keys. The two modes must not be blended into one score.

## Results

Archived in [`results/v1/`](results/v1/): every raw response, `metadata.json` (versions,
device, data and prompt hashes) and an audited `summary.md` per configuration.

- **Checkpoint revision:** `convaiinnovations/laya` @ `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`
  (`--revision`)
- **Hardware:** CPU, Windows, Python 3.10; recorded 2026-09-29
- **Protocol:** 64 cases x 2 prompt modes x 3 repeats per configuration. Quality is scored on
  the first repeat. All three repeats gave identical predictions in every configuration, and an
  earlier unpinned run gave the same accuracy figures.

Reproduce with:

```bash
python run_all.py --revision 55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851
```

**Persian instructions (`choice_fa`)**

| | multilingual | multilingual + norm | english |
|---|---|---|---|
| **overall** | **36/64** | **37/64** | **21/64** |
| formal | 6/8 | 5/8 | 3/8 |
| colloquial | 5/8 | 5/8 | 2/8 |
| finglish | 4/8 | 4/8 | 4/8 |
| orthography | 4/8 | 5/8 | 2/8 |
| code_mixed | 5/8 | 5/8 | 2/8 |
| negation | 3/8 | 3/8 | 2/8 |
| sarcasm_taarof | 5/8 | 5/8 | 3/8 |
| digits | 4/8 | 5/8 | 3/8 |
| false `cancel` | 4 | 3 | **38** |
| macro-F1 | 0.513 | 0.513 | 0.261 |
| ECE | 0.273 | 0.257 | 0.105 |
| p50 latency (CPU)* | 218 ms | 696 ms | 603 ms |

**English instructions (`choice_en`)**

| | multilingual | multilingual + norm | english |
|---|---|---|---|
| **overall** | **41/64** | **40/64** | **29/64** |
| formal | 5/8 | 5/8 | 4/8 |
| colloquial | 7/8 | 6/8 | 3/8 |
| finglish | 4/8 | 4/8 | 5/8 |
| orthography | 5/8 | 5/8 | 4/8 |
| code_mixed | 6/8 | 6/8 | 5/8 |
| negation | 3/8 | 3/8 | 2/8 |
| sarcasm_taarof | 6/8 | 6/8 | 2/8 |
| digits | 5/8 | 5/8 | 4/8 |
| false `cancel` | 2 | 3 | 16 |
| macro-F1 | 0.633 | 0.619 | 0.396 |
| ECE | 0.250 | 0.202 | 0.084 |
| p50 latency (CPU)* | 170 ms | 543 ms | 1467 ms |

Random guessing scores 16/64 on this balanced set.

\* Latency was measured on a shared desktop machine and varied by up to several times between
runs of the same configuration. Treat it as indicative only: this is not a speed benchmark.
Normalisation itself costs microseconds, so its latency differences are noise.

### Observations

- **The English checkpoint does not read Persian.** With Persian instructions it predicted
  `cancel` for 53 of 64 messages, which gives 38 false cancellations. This is consistent with
  Laya's own advice to route non-Latin scripts to `laya-multilingual`. Its low ECE reflects
  low confidence, not good calibration of useful answers. At load it also warned that some
  shipped temperatures were clamped.
- **`laya-multilingual` is well above chance but not production-ready zero-shot.** It scored
  36-41/64 with a mean answer confidence of about 83%, so it is over-confident (ECE 0.25-0.27).
- **English instructions scored higher than Persian ones** (41 vs 36/64 on the same Persian
  messages). With 64 cases this 5-case gap is suggestive, not established.
- **`other` is rarely chosen.** Under Persian instructions only 2 of 16 `other` messages were
  labelled `other`, and thanks, suggestions and general questions went to `billing` or
  `technical`. Also, 8 of 16 `cancel` messages were labelled `technical`.
- **Negation is the weakest family** (at most 3/8 in every configuration).
- **Normalisation had no measurable effect** (±1 case). The errors do not come mainly from
  Arabic ي/ك, digit forms or ZWNJ.

These are regression diagnostics on 64 synthetic cases. They are not a model ranking and
should not be compared directly with other benchmarks' scores. Do not tune prompts or
checkpoints on these cases and then report them as held out.

## Files

| path | purpose |
|---|---|
| `data/cases.jsonl` | frozen messages, reference labels, rationale, family, trap tag |
| `data/manifest.json` | hashes, counts, label policy, provenance, reviewer |
| `prompts.py` | the two frozen prompt modes |
| `normalize.py` | optional Persian normalisation (ablation) |
| `run.py` | runner: `laya` backend or the `keyword` pipeline check |
| `audit.py`, `metrics.py`, `common.py` | model-free validation and scoring |
| `compare.py`, `run_all.py` | side-by-side tables; the full study in one command |
| `tools/update_manifest.py` | rehash after an intentional data or prompt change |
| `tests/` | model-free tests, including corruption and hash-mismatch checks |

## Label policy

The label is the department that has to act on the message's **main request**. A topic that
is mentioned only to be negated does not set the label. Sarcastic thanks is labelled by the
complaint it carries; genuine thanks is `other`. Stopping auto-renewal or closing or deleting
the account is `cancel`. The label policy is also in `data/manifest.json`.

## Limitations

- Only 64 related synthetic examples, with a single reviewer. There is no multi-annotator
  agreement yet.
- Results depend on checkpoint, revision, dtype, device and prompt wording. Treat them as a
  regression diagnostic for Persian, not as evidence about other languages or tasks.
- Finglish transliteration and the colloquial or formal split are simplified. Real traffic is
  noisier.
- Persian fine-tuning is a separate question that this benchmark does not answer. It gives a
  fine-tune something to be measured against.

## License

MIT (see `LICENSE`). It does not change Laya's Apache-2.0 license.
