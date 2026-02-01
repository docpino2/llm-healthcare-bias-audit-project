# LLM Healthcare Bias Audit — Dataset Scaffold

This repository contains a reproducible dataset scaffold for a counterfactual audit of Large Language Models (LLMs)
used in healthcare decision-making.

## Study design (RQ2 focus)

- Base cases: 50 neutral clinical vignettes
- Demographic factors:
  - Gender: Male / Female
  - Ethnicity: Caucasian / African American / Hispanic / Asian
  - Age group: Young / Adult / Elderly
- Models (default): 4

Total inferences: 50 × 2 × 4 × 3 × 4 = **4,800**

## Files

### `data/processed/base_cases.csv`
One row per neutral vignette (no demographics injected).

### `data/processed/demographics.csv`
All counterfactual demographic combinations (24 rows).

### `data/processed/inferences_skeleton.csv`
The full Cartesian product (case × demographics × model), with standardized prompts pre-generated.
Fields for `raw_response` and post-processing outputs are included as empty columns.

## Prompt format

Prompts are standardized to:
1) Diagnosis (single best)
2) Disposition: Home / Additional Tests or Clinic Follow-up / Emergency Department
3) Brief rationale (2–4 sentences)
4) Risk level: Low / Medium / High

## Regenerating the inference skeleton

```bash
python src/generate_inference_skeleton.py \
  --base-cases data/processed/base_cases.csv \
  --demographics data/processed/demographics.csv \
  --out data/processed/inferences_skeleton.csv \
  --models "GPT-5.2" "Gemini 3.0" "Claude 3.5" "DeepSeek 3.2"
```

## Notes

- The original vignette source file used in this build was `vignettes-2015.txt` from the public clinical vignette repository.
- Five additional vignettes were added to reach exactly 50 base cases. These are marked as `source = author-added (to reach n=50)` in `base_cases.csv`.
