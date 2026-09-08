# ai-agent-security-decoy-metric

Working note and attack code for the Kaggle competition **AI Agent Security — Multi-Step Tool Attacks** (OpenAI · Google · IEEE).

The central result: under the competition's permissive *public* guardrail, the attack score reduces almost entirely to throughput — the number of independent, predicate-firing tool-call candidates that fit a fixed replay budget. The *private* leaderboard, revealed after the competition closed, turns out to be a different game: only the guardrail-agnostic confused-deputy action transfers, so a high public score can be a decoy for a private score of zero.

The note derives the scoring stack from the SDK source, documents a battery of refuted levers and the mechanism that kills each, retracts two of its own earlier findings that did not survive re-verification at higher sample sizes, and — after close — reproduces the ~40 private frontier firsthand by replaying published top-team code unchanged.

## The note

The full working note, with all 25 figures, is in [`working_note.md`](working_note.md).

Competition: https://www.kaggle.com/competitions/ai-agent-security-multi-step-tool-attacks

## Results

Our own submissions:

| Board | Score | Note |
|---|---|---|
| Public (OptimalGuardrail) | 89.55 | single-post exfiltration + reasoning-suppression forge |
| Private ceiling | ~27.9 | confused-deputy only — the one predicate that transfers to the held-out guardrail |

### Control experiment

To test *why* our private score stops at ~27.9, we replayed two other teams' **published** attacks unmodified,
purely as a measurement instrument. Those scores are theirs, not claimed here, and their code is not
redistributed (sources linked below) — the value is the lesson:

- Replaying the published 5th-place attack returns private ~40.3 (n=14). Lesson: the frontier is a
  throughput-*engineering* gap, not a reachability one — our ceiling was an engineering limit, not a
  conceptual one.
- Replaying a public-heavy attack returns 57.1 public / 0.00 private. Lesson: a top-decile *public* score can
  transfer to *zero* on the held-out guardrail — the decoy, at its sharpest.

## Layout

```
attack.py                        the attack algorithm (AttackAlgorithm submission engine)
working_note.md                  the full working note
figures/                         figures referenced by the note
research/
  cv_real.py                     research harness — runs each measurement, writes the *_results.csv tables
  analyze_private_campaign.py    private-board analysis
reproduce/
  make_figures.py                regenerates the schematic figures (writes ./*.png)
  analyze_v100_results.py        regenerates the experiment figures from the *_results.csv tables
  *_results.csv                  the raw per-experiment measurement tables behind every experiment figure
data/
  my_submissions.csv             every submission we made, with public and private scores
  campaign_scores.csv            the post-close private-board sweep
  private_ledger_summary.json    summary of what scored on the private guardrail
  batch*_variants.json           the variant configurations submitted in each batch
tools/
  submit_gpu_variant.py          push a parameterized GPU kernel and submit it
  submit_external.py             replay an external attack.py for firsthand reproduction
```

## Reproducing the figures

The pipeline is end-to-end our own: `attack.py` produces submissions, `cv_real.py` runs each measurement
and writes the `*_results.csv` tables, and `reproduce/` turns those tables into the figures.

```bash
pip install -r requirements.txt
cd reproduce
python make_figures.py            # schematic figures (fig1–fig11), written next to the script
python analyze_v100_results.py    # experiment figures (fig12–fig24) from the *_results.csv tables
```

## Running

`attack.py` runs inside a Kaggle notebook against the competition SDK (`aicomp_sdk` / `kaggle_evaluation`), which the competition provides in the notebook environment. The research and tooling scripts run locally and read Kaggle credentials from the environment:

```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_key            # or place them in ~/.kaggle/kaggle.json
pip install -r requirements.txt
```

No credentials are stored in this repository.

## Reproduced work

The firsthand reproductions in the note replay other teams' *published* attack code, unmodified. That code is **not** redistributed here — see the original sources:

- 5th place (Giovanny): https://github.com/dreuxx/Kaggle-AI-Agent-Security-5th-Place-Solution
- A public confused-deputy attack: https://github.com/Okpeyemi/ai-agent-security-attacks

## Responsible use

Every construction here is confined to the competition's deterministic offline sandbox and its synthetic fixtures (fake secrets such as `TOKEN=admin123`). No real credential, recipient, or network target is involved anywhere. Techniques are reported so the corresponding defenses can be built; the note's final section gives the mitigation that closes each one.

## License

MIT — see [LICENSE](LICENSE).
