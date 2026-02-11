# Do LLMs Differentiate Epistemic Belief from Non-Epistemic Belief?

## Overview
This research investigates whether large language models (LLMs) behaviorally differentiate between **epistemic beliefs** (evidence-based, truth-tracking) and **non-epistemic beliefs** (value-based, desire-based, faith-based), inspired by Vesga et al.'s argument that humans maintain categorically different types of beliefs.

## Key Findings

- **Perfect classification**: GPT-4.1 classifies 40 belief statements as epistemic vs. non-epistemic with 100% accuracy (Cohen's κ = 1.0)
- **Radically different response strategies**: The model spontaneously uses evidence 100% of the time for epistemic beliefs but only 3.3% for non-epistemic beliefs (Cramér's V = 0.95)
- **Counterintuitive revision asymmetry**: The model resists revising well-established factual beliefs *more* than value beliefs (revision strength 2.55 vs. 3.27, p = 0.002), reflecting strong epistemic priors for established facts
- **Factive verb sensitivity**: "Alex knows X" implies truth 63% of the time; "Alex believes X" implies truth 0% (Kruskal-Wallis p < 0.0001)
- **Bottom line**: LLMs are not "epistemically flat" — they maintain a functional distinction between belief types analogous to the one in human cognition

## Reproduction

### Setup
```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate
uv add openai numpy scipy matplotlib pandas seaborn

# Set API key
export OPENAI_API_KEY="your-key-here"
```

### Run experiments
```bash
cd src
python run_experiments.py  # Runs all 4 experiments (~47 min)
python analyze_results.py  # Generates analysis and plots
```

## File Structure
```
├── REPORT.md                  # Full research report with all results
├── README.md                  # This file
├── planning.md                # Research plan and motivation
├── literature_review.md       # Comprehensive literature review (29 papers)
├── resources.md               # Catalog of datasets, papers, code
├── src/
│   ├── stimuli.py             # Belief stimuli (20 epistemic, 20 non-epistemic)
│   ├── run_experiments.py     # Main experiment runner (4 experiments)
│   └── analyze_results.py     # Statistical analysis and visualization
├── results/
│   ├── config.json            # Experiment configuration
│   ├── exp1_classification.json
│   ├── exp2_differential_response.json
│   ├── exp3_belief_revision.json
│   ├── exp4_factive_verb.json
│   ├── summary_stats.json
│   └── plots/
│       ├── exp1_classification.png
│       ├── exp2_differential_response.png
│       ├── exp3_belief_revision.png
│       ├── exp4_factive_verb.png
│       └── summary_all_experiments.png
├── papers/                    # 30 downloaded research papers (PDFs)
├── datasets/                  # Downloaded benchmark datasets (KaBLE, MindGames, etc.)
└── code/                      # Cloned baseline repositories
```

## Full Report
See [REPORT.md](REPORT.md) for the complete research report including methodology, statistical analysis, visualizations, discussion, and limitations.
