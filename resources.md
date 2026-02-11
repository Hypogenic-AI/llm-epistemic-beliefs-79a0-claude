# Resources Catalog

## Research Question
**Do LLMs differentiate epistemic belief from non-epistemic belief?**

---

## 1. Papers (30 downloaded PDFs)

All papers are stored in `papers/` with naming convention `{arxiv_id}_{descriptive_name}.pdf`.
Paper-to-filename mapping is in `papers/arxiv_ids.json`.

### Tier 1: Most Directly Relevant (Deep-Read)

| arXiv ID | Title | Authors | Year | Key Contribution |
|----------|-------|---------|------|-----------------|
| 2410.21195 | Belief in the Machine: Investigating Epistemic Reasoning in Language Models | Suzgun et al. | 2024 | KaBLE benchmark (13K questions); shows LLMs fail to distinguish belief from knowledge |
| 2405.21030 | Standards for Belief Representations in LLMs | Herrmann & Levinstein | 2024 | Four criteria (accuracy, coherence, uniformity, use) for genuine belief representations |
| 2402.18496 | Language Models Represent Beliefs of Self and Others | Bortoletto et al. | 2024 | Probing reveals separate linear representations for self vs. other beliefs in Mistral-7B |
| 2505.14685 | Language Models Use Lookbacks to Track Beliefs | Lanham et al. | 2025 | Mechanistic discovery of pointer-dereference attention pattern for belief tracking |
| 2511.00617 | Belief Dynamics Reveal the Dual Nature of ICL and Activation Steering | Bigelow et al. | 2025 | Bayesian framework: ICL = evidence accumulation (epistemic), steering = prior change (non-epistemic) |
| 2511.22746 | Epistemic Fragility in LLMs | Krastev et al. | 2025 | Prompt framing systematically modulates misinformation correction in frontier LLMs |
| 2511.19166 | Representational and Behavioral Stability of Truth | Dies et al. | 2026 | P-StaT framework; epistemic familiarity governs belief stability |
| 2305.03353 | MindGames: Targeting ToM with Dynamic Epistemic Modal Logic | Sileo & Lernould | 2023 | DEL-based epistemic reasoning benchmark; LLMs near chance on formal epistemic logic |

### Tier 2: Theory of Mind and Belief Benchmarks

| arXiv ID | Title | Key Contribution |
|----------|-------|-----------------|
| 2302.02083 | Evaluating LLMs for Theory of Mind | Systematic ToM evaluation |
| 2306.00924 | Minding LM Lack of ToM | Documents systematic ToM failures |
| 2310.03051 | How Far Are LLMs from ToM Agents? | Comprehensive ToM assessment |
| 2407.06004 | From Perceptions to Beliefs in ToM | Perceptual grounding to belief formation |
| 2310.16755 | Hi-ToM Benchmark | Higher-order ToM with depth scaling |
| 2109.14723 | BeliefBank | Structured belief storage and consistency |
| 2310.15421 | FANToM Benchmark | False beliefs in conversational contexts |
| 2502.06470 | Think Twice: Perspective-Taking ToM | Perspective-taking for ToM |
| 2410.13648 | SimpleToM: Exposing the Gap | Mental state understanding vs. behavioral prediction |
| 2501.15355 | LLMs ToM via Counterfactual Reflection | Counterfactual reasoning for ToM |

### Tier 3: Calibration, Bayesian Reasoning, and Supporting Papers

| arXiv ID | Title | Key Contribution |
|----------|-------|-----------------|
| 2408.12022 | Epistemic Language and Bayesian ToM | Bayesian approach to epistemic language |
| 2410.16270 | Reflection Bench: Epistemic | Epistemic reflection benchmark |
| 2501.05032 | Representations of Fact, Fiction, Forecast | Factual vs. fictional content representations |
| 1603.07704 | Probabilistic Coherence and Bayesian LLM | Bayesian coherence in language models |
| 2504.19622 | Evidence to Belief: Bayesian | Bayesian evidence integration |
| 1910.07514 | Reducing Overconfidence and Calibration | Calibration and epistemic confidence |
| 2510.09033 | LLMs Know What Humans Know | Modeling human knowledge attribution |
| 2407.15814 | Perceptions of Linguistic Uncertainty | Uncertainty expression processing |
| 2309.02144 | Belief Revision in LLMs | Belief updating with contradictory evidence |
| 2408.07237 | Semantic Embedding of Beliefs | Embedding-level belief representations |

---

## 2. Datasets

All datasets are stored in `datasets/` directory.

### Primary Datasets for Experimentation

| Dataset | Location | Size | Format | Source | Relevance |
|---------|----------|------|--------|--------|-----------|
| **KaBLE** | `datasets/kable/` | 13,000 examples (13 JSONL files) | JSONL | [GitHub](https://github.com/suzgunmirac/belief-in-the-machine) | Core benchmark for belief/knowledge distinction |
| **MindGames** | `datasets/mindgames/` | 3,725 test examples | JSONL | [HuggingFace](https://huggingface.co/datasets/sileod/mindgames) | Formal epistemic reasoning (DEL-based) |
| **Trilemma of Truth** | `datasets/trilemma_of_truth/` | 12,565 examples (3 configs) | JSONL | [HuggingFace](https://huggingface.co/datasets/carlomarxx/trilemma-of-truth) | True/False/Synthetic statements for stability testing |
| **Representational Stability (Fictional)** | `datasets/representational_stability/` | 3,952 examples | JSONL | [HuggingFace](https://huggingface.co/datasets/samanthadies/representational_stability) | Fictional statements for epistemic familiarity testing |

### Supporting Datasets

| Dataset | Location | Size | Format | Source | Relevance |
|---------|----------|------|--------|--------|-----------|
| **ToMi-NLI** | `datasets/tomi_nli/` | 17,982 examples (3 splits) | Arrow | [HuggingFace](https://huggingface.co/datasets/alisawuffles/tomi-nli) | Theory of Mind NLI tasks |
| **SimpleToM** | `datasets/simpletom/` | 4,588 examples (4 configs) | Arrow | [HuggingFace](https://huggingface.co/datasets/allenai/SimpleToM) | Mental state, behavior, and judgment QA |
| **Theory of Mind** | `datasets/theory_of_mind/` | 539 examples | Arrow | [HuggingFace](https://huggingface.co/datasets/grimulkan/theory-of-mind) | ToM instruction-response pairs |
| **Epistemic Clarification** | `datasets/epistemic_clarification/` | 20 examples | - | HuggingFace | Too small for main experiments |

### Dataset Details

#### KaBLE (13 JSONL files, 1000 entries each)
- Tasks: verification-of-fact, verification-of-belief, confirmation-of-first-person-belief, confirmation-of-third-person-belief, confirmation-of-first-person-knowledge, confirmation-of-third-person-knowledge, recursive tasks
- Fields: `experiment_setup`, `subject`, `idx`, `type` (factual/false), `raw_sentence`, `query`, `answer`
- 10 disciplines: Math, Physics, Chemistry, Biology, Medicine, Psychology, History, Geography, Law, Economics

#### MindGames
- Fields: `premise`, `smcdel_problem`, `n_announcements`, `hypothesis`, `setup`, `hypothesis_depth`, `n_agents`, `label`, `names`
- 4 setups: forehead, forehead-mirror, thirst, explicit
- Labels: entailment / not_entailment

#### Trilemma of Truth (3 configs)
- `city_locations`: 3,999 examples
- `med_indications`: 3,849 examples
- `word_definitions`: 4,717 examples
- Statement types: True, False, Synthetic (fabricated entities)
- Fields: `statement`, `object_1`, `object_2`, `correct`, `negation`, `real_object`, `fake_object`

---

## 3. Code Repositories

All repositories are cloned into `code/` directory.

| Repository | Location | Source | Description |
|------------|----------|--------|-------------|
| **belief-in-the-machine** | `code/belief-in-the-machine/` | [GitHub](https://github.com/suzgunmirac/belief-in-the-machine) | KaBLE benchmark evaluation code + dataset |
| **representational_stability** | `code/representational_stability/` | [GitHub](https://github.com/samanthadies/representational_stability) | P-StaT framework for truth stability analysis |
| **llm-theory-of-mind** | `code/llm-theory-of-mind/` | [GitHub](https://github.com/sileod/llm-theory-of-mind) | MindGames DEL problem generation and evaluation |

### Additional Code Resources (Not Cloned)

| Resource | URL | Description |
|----------|-----|-------------|
| RepBelief | https://walter0807.github.io/RepBelief/ | Belief probing code from "LM Represent Beliefs" paper |
| Belief tracking (CausalToM) | https://belief.baulab.info | Lookback mechanism analysis code |
| Anthropic Persona Evals | https://github.com/anthropics/evals/tree/main/persona | Persona datasets for belief dynamics experiments |
| nnsight | https://github.com/ndif-team/nnsight | Neural network inspection tool used in several papers |

---

## 4. Search Results

Raw paper search results are stored in `paper_search_results/` as JSONL files from multiple queries:
- `epistemic_belief_language_models_cognition_*.jsonl`
- `LLM_belief_representation_theory_of_mind_*.jsonl`
- `LLM_epistemic_reasoning_uncertainty_calibration_*.jsonl`
- `belief_tracking_natural_language_understanding_factive_non-factive_verbs_*.jsonl`

---

## 5. Recommended Experiment Priorities

### Priority 1: Direct Tests of Epistemic vs. Non-Epistemic Distinction
1. **KaBLE replication and extension**: Run KaBLE benchmark on current frontier models; extend with non-epistemic belief items (desires, values, cultural beliefs)
2. **MindGames epistemic reasoning**: Test whether models distinguish "know that" from "believe that" in formal epistemic logic contexts

### Priority 2: Internal Representation Analysis
3. **Belief probing**: Apply RepBelief-style probing to test whether internal representations distinguish epistemic from non-epistemic belief attributions
4. **P-StaT epistemic vs. non-epistemic stability**: Test whether epistemic beliefs (evidence-based) are more stable under perturbation than non-epistemic beliefs (value-based, desire-based)

### Priority 3: Behavioral Robustness
5. **Epistemic fragility with belief types**: Extend Krastev et al.'s methodology to compare how prompt framing affects epistemic vs. non-epistemic assertions
6. **Belief dynamics decomposition**: Use Bigelow et al.'s framework to test whether the evidence/prior decomposition maps onto epistemic/non-epistemic categories

### Key Models to Test
- Open-source (for probing): Llama-3.1-8B, Mistral-7B, Qwen-2.5-7B/14B
- Frontier (for behavioral): Claude, GPT-4/5, Gemini
