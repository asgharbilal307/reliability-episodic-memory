# Reliability-Aware Episodic Memory for Self-Improving LLM Agents

**Course:** Data Science — FAST-NUCES Lahore, 2026  
**Author:** Bilal Asghar  
**Target Venues:** EMNLP Findings / AAMAS 2027 / NeurIPS LLM Agents Workshop

---

## Problem Statement

Most LLM agent pipelines are stateless: an agent reasons well within a single session but retains no persistent knowledge of past successes or failures. Recent work (Reflexion, ExpeL, MemGPT) shows that verbal reflections stored in external memory can improve agent performance over repeated tasks. However, all existing episodic memory systems share a critical flaw: **they retrieve memories purely by semantic similarity**.

Similarity is a proxy for relevance, not reliability. Consider two retrieved memories for a new task:

| Reflection | Similarity | Past success rate |
|---|---|---|
| *Try reversing the list first* | 0.95 | 10% |
| *Check the loop boundary condition* | 0.85 | 90% |

Every current system retrieves Memory A. A reliable agent should retrieve Memory B.

**Research question:** Can a reliability-aware retrieval policy — weighting memories by historical success rate, recency, and estimated confidence alongside semantic similarity — demonstrably improve agent performance over similarity-only retrieval?

---

## Dataset Details

### Task Dataset: `data/humaneval_subset.json`

| Property | Value |
|---|---|
| Source | HumanEval (Chen et al., 2021) — hand-curated subset |
| Size | 80 problems |
| Target variable | Binary task success (passing / failing pytest) |
| Features | `task_id`, `bug_category`, `prompt` (buggy code), `canonical_solution`, `test`, `function_length`, `complexity` |
| Bug categories | `off_by_one` (20), `type_error` (20), `scope_error` (20), `logic_error` (20) |
| Class balance | Perfectly balanced — 20 problems per category |

> Dataset is included in the repo. Original HumanEval benchmark: https://github.com/openai/human-eval

### Memory Bank: `data/synthetic_memories.json`

| Property | Value |
|---|---|
| Purpose | Schema validation and retrieval policy sanity checking before Phase 2 agent runs |
| Size | 25 hand-crafted memories |
| Fields | `reflection`, `task_category`, `success_rate`, `times_retrieved`, `times_helpful`, `confidence`, `recency_ts` |
| Reliability split | 20 high-SR (>= 0.7), 5 low-SR (< 0.3) |

---

## Cleaning Steps

1. **Missing values:** None found across all 80 problems and 25 memories.
2. **Duplicate IDs:** None found.
3. **Empty tests:** None found.
4. **Control problems:** 46 of 80 problems have identical `prompt` and `canonical_solution` (intentionally included to test the agent's ability to recognise already-correct code).
5. **Recency normalisation:** `recency_ts` (Unix timestamp) converted to `recency_score` = exp(-0.01 · Δt days) in `data/processed/memories_preprocessed.json`.
6. **Range validation:** All reliability fields (`success_rate`, `confidence`, `recency_score`) verified in [0, 1]. `times_helpful ≤ times_retrieved` for all memories.

---

## Key EDA Findings

| Finding | Implication |
|---|---|
| Dataset perfectly balanced (20 per category) | Per-category success rates in Phase 2 are directly comparable without reweighting |
| Mean function length 4.7 lines (range 2–16) | Low token cost per episode; 100-episode runs feasible on Groq free tier |
| Logic errors and scope errors have higher cyclomatic complexity | Expected to benefit more from reliability-aware retrieval; tested in per-category analysis |
| Moderate correlation between length and complexity (r ≈ 0.6) | Both features retained as independent predictors |
| 5 low-SR memories get high proxy similarity | Similarity-vs-reliability gap exists in synthetic bank; validates experimental design |
| Confidence and SR highly correlated on synthetic data | Calibration design is correct; real ECE will be the Phase 2 result |
| Recency score retains 74% at 30 days (λ=0.01) | Decay is slow enough for 100-episode runs without premature expiry |

### Visualisations

| File | Description |
|---|---|
| `images/fig1_bug_distribution.png` | Bar chart of problem count per bug category — confirms perfect balance |
| `images/fig2_function_lengths.png` | Histograms of function length per category — shows logic errors are longest |
| `images/fig3_complexity_boxplot.png` | Boxplots of cyclomatic complexity per category — scope and logic errors are more complex |
| `images/fig4_length_vs_complexity.png` | Scatter plot of length vs complexity — confirms moderate positive correlation |
| `images/fig5_memory_sr_dist.png` | Histogram of success rates in synthetic memory bank — 20 high-SR, 5 low-SR |
| `images/fig6_calibration_preview.png` | Confidence vs SR scatter — confirms synthetic bank is well-calibrated by design |
| `images/fig7_recency_decay.png` | Recency decay curve for λ=0.01 — shows decay rate over 180 days |
| `images/fig8_sr_confidence_heatmap.png` | Heatmap of mean SR and confidence by category — consistent across categories |

---

## How to Run

```bash
# 1. Clone and install dependencies
git clone <repo-url>
cd reliability-episodic-memory
pip install -r requirements.txt

# 2. Run notebooks in order
jupyter notebook notebooks/01_data_curation.ipynb
jupyter notebook notebooks/02_eda.ipynb
jupyter notebook notebooks/03_preprocessing.ipynb
```

Processed outputs land in `data/processed/`. All notebook cells must be run in order; outputs are pre-rendered in the committed versions.

---

## Repository Structure

```
reliability-episodic-memory/
├── README.md
├── requirements.txt
├── data/
│   ├── humaneval_subset.json         # 80-problem task dataset
│   ├── synthetic_memories.json       # 25-memory bank for schema validation
│   └── processed/
│       ├── problems_clean.json       # Cleaned dataset with split field
│       └── memories_preprocessed.json # Memories with recency_score computed
├── notebooks/
│   ├── 01_data_curation.ipynb        # Dataset loading, test runner validation, schema checks
│   ├── 02_eda.ipynb                  # Full EDA with 8 visualisations
│   └── 03_preprocessing.ipynb       # Normalisation, composite score validation, exports
└── images/
    ├── fig1_bug_distribution.png
    ├── fig2_function_lengths.png
    ├── fig3_complexity_boxplot.png
    ├── fig4_length_vs_complexity.png
    ├── fig5_memory_sr_dist.png
    ├── fig6_calibration_preview.png
    ├── fig7_recency_decay.png
    └── fig8_sr_confidence_heatmap.png
```

---

## References

1. Shinn et al., *Reflexion: Language Agents with Verbal Reinforcement Learning*, NeurIPS 2023.
2. Zhao et al., *ExpeL: LLM Agents Are Experiential Learners*, AAAI 2024.
3. Packer et al., *MemGPT: Towards LLMs as Operating Systems*, arXiv 2023.
4. Chen et al., *Evaluating Large Language Models Trained on Code*, arXiv 2021.
5. Kadavath et al., *Language Models (Mostly) Know What They Know*, arXiv 2022.
