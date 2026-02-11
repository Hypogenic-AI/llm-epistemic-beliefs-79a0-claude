# Literature Review: Do LLMs Differentiate Epistemic Belief from Non-Epistemic Belief?

## 1. Introduction and Scope

This review synthesizes the current literature on whether large language models (LLMs) can differentiate between **epistemic belief** (beliefs grounded in evidence, truth-tracking, and justification -- e.g., "I know that water boils at 100C") and **non-epistemic belief** (beliefs driven by desires, values, social context, cultural commitments, or mere acceptance -- e.g., "I believe in fairness"). The distinction is fundamental in epistemology: knowledge requires justified true belief, whereas non-epistemic beliefs need not be truth-apt at all.

The review covers 29 papers spanning three interconnected research threads: (1) behavioral benchmarks testing whether LLMs distinguish belief, knowledge, and fact; (2) mechanistic and representational studies probing how LLMs internally encode belief states; and (3) stability and robustness studies examining whether LLM belief-like behavior is anchored in epistemic content or driven by non-epistemic contextual features.

---

## 2. Core Findings: LLMs Struggle to Distinguish Belief from Knowledge

### 2.1 The KaBLE Benchmark (Suzgun et al., 2024)

**Paper:** "Belief in the Machine: Investigating Epistemic Reasoning in Language Models" (arXiv:2410.21195)

The most directly relevant study. The Knowledge and Belief in Language Evaluation (KaBLE) benchmark consists of 13,000 questions across 13 tasks organized into three categories:

- **Verification tasks**: Can the model assess whether a factual/false statement is true?
- **Belief confirmation tasks**: Can the model reason about what an agent believes vs. knows?
- **Recursive knowledge tasks**: Can the model handle nested epistemic states ("A knows that B believes...")?

**Key results across 15 LLMs (including GPT-4, Claude-3, Llama-3):**
- Models achieve 85.7% accuracy on factual belief confirmation but only 54.4% on false belief confirmation, revealing a systematic inability to represent beliefs that diverge from reality.
- **First-person vs. third-person asymmetry**: Models score 54.4% on first-person belief tasks (e.g., "Do I believe X?") but 80.7% on third-person tasks ("Does Alice believe X?"). This suggests models cannot genuinely adopt a believer's perspective when it conflicts with world knowledge.
- Models fail to respect the **factive nature of knowledge**: they do not consistently distinguish "X knows P" (which entails P is true) from "X believes P" (which does not).
- Over-reliance on linguistic cues rather than epistemic reasoning drives many correct answers.

**Implication for the research question**: LLMs do not robustly differentiate epistemic belief (knowledge) from mere belief. Their performance degrades precisely at the boundary where this distinction matters most -- when beliefs are false or when first-person perspective requires maintaining a belief state that contradicts the model's factual knowledge.

### 2.2 Standards for Belief Representations (Herrmann & Levinstein, 2024)

**Paper:** "Standards for Belief Representations in LLMs" (arXiv:2405.21030)

A theoretical paper proposing four criteria that LLM belief representations must satisfy to count as genuine beliefs:

1. **Accuracy**: Belief representations should track truth (the model's "beliefs" should be mostly correct).
2. **Coherence**: Beliefs should be logically consistent with each other.
3. **Uniformity**: The same content should be represented consistently across different contexts and prompting conditions.
4. **Use**: Beliefs should actually guide the model's behavior and outputs.

The paper argues that current LLMs likely fail on **uniformity** (belief expression changes with prompting) and **use** (internal representations may not drive outputs). This framework provides a principled basis for evaluating whether observed LLM behavior constitutes genuine epistemic belief or merely contextually-triggered pattern matching.

---

## 3. Internal Representations of Belief

### 3.1 Separate Representations for Self vs. Other Beliefs (Bortoletto et al., 2024)

**Paper:** "Language Models Represent Beliefs of Self and Others" (arXiv:2402.18496)

Using probing experiments on Mistral-7B with the BigToM dataset, this study found:
- **Separate linear representations** for the model's own beliefs vs. attributed beliefs of characters in stories.
- Logistic regression probes achieve high accuracy in classifying belief states from intermediate layer activations.
- **Activation interventions** (modifying internal representations) can causally alter the model's belief attributions, confirming that the representations are not merely correlational.
- The representations are **linearly separable**, suggesting the model maintains distinct computational pathways for self-belief and other-belief.

**Relevance**: This provides evidence that LLMs do maintain some internal differentiation between belief types -- at minimum, between "what I believe" and "what another agent believes." However, this is a self/other distinction, not an epistemic/non-epistemic distinction per se.

### 3.2 Lookback Mechanism for Belief Tracking (Lanham et al., 2025)

**Paper:** "Language Models Use Lookbacks to Track Beliefs" (arXiv:2505.14685)

A mechanistic interpretability study on Llama-3-70B that discovered a **lookback mechanism** -- a pointer-dereference computational pattern in transformer attention heads that implements belief tracking:

- When processing Theory of Mind scenarios, specific attention heads "look back" to the point in the narrative where a character's belief was formed, effectively implementing a form of epistemic memory.
- The mechanism operates through **sparse, interpretable circuits** rather than distributed representations.
- Created the **CausalToM** dataset for evaluating causal mechanisms of belief tracking.
- The mechanism handles both true and false belief scenarios, but with reduced reliability for false beliefs.

**Relevance**: Demonstrates that LLMs have developed specific computational mechanisms for tracking epistemic states. However, the mechanism tracks *what* an agent believes rather than *whether* the belief is epistemically justified -- the distinction between epistemic and non-epistemic belief is not directly represented in this mechanism.

### 3.3 Belief Dynamics as Bayesian Inference (Bigelow et al., 2025)

**Paper:** "Belief Dynamics Reveal the Dual Nature of In-Context Learning and Activation Steering" (arXiv:2511.00617)

This paper formalizes LLM belief dynamics through a Bayesian framework, showing that:
- **In-context learning (ICL)** operates as **evidence accumulation** (epistemic updating through the likelihood function).
- **Activation steering** operates as **prior modification** (non-epistemic disposition change).
- These two mechanisms are **additive in log-odds space** and produce predictable phase transitions in behavior.
- The unified model achieves r=0.98 correlation with actual LLM behavior across five persona domains.

**Key implication for the research question**: This paper provides the clearest evidence that LLMs have **two separable pathways** that map onto the epistemic/non-epistemic distinction: evidence-based updating (epistemic) and disposition-based modification (non-epistemic). The fact that these are additive and independently controllable suggests LLMs do maintain some form of this distinction, at least at the mechanistic level.

---

## 4. Stability and Robustness of LLM Beliefs

### 4.1 Epistemic Fragility (Krastev et al., 2025)

**Paper:** "Epistemic Fragility in LLMs: Prompt Framing Systematically Modulates Misinformation Correction" (arXiv:2511.22746)

Tested 4 frontier LLMs (Claude Sonnet 4.5, ChatGPT-5, Grok-4, Gemini 2.5 Pro) across 320 misinformation prompts varying in open-mindedness, user intent, user role, and complexity:

- **Creative intent** reduces correction odds by 89% relative to information-seeking intent (OR=0.11).
- **Assertive expert** role reduces correction by 21% compared to naive inquirer.
- **Open framing** increases correction by 75% over closed framing.
- Claude Sonnet 4.5 showed strongest corrections (mean stance 8.40/11); Gemini 2.5 Pro weakest (5.77/11).

**Critical finding**: The same factual claim receives systematically different treatment depending on **non-epistemic features** of the prompt (tone, social role, creative vs. informational intent). This demonstrates that LLMs' truth-assertion behavior is modulated by non-epistemic contextual factors, suggesting they do not maintain a stable epistemic commitment independent of social/conversational context.

### 4.2 Representational Stability of Truth (Dies et al., 2026)

**Paper:** "Representational and Behavioral Stability of Truth in Large Language Models" (arXiv:2511.19166)

Introduced the **P-StaT** (Perturbation Stability of Truth) framework, testing 16 open-source LLMs across three factual domains:

- **Epistemic familiarity governs stability**: Synthetic (unfamiliar, fabricated) content destabilizes beliefs far more than fictional (familiar, from known stories) content.
  - Word Definitions domain: 32.7% epistemic retractions under synthetic perturbation vs. much lower under fictional perturbation.
- **Representational vs. behavioral dissociation**: At the representational level (probing), synthetic content clusters near factual content in activation space, making it harder for the model to distinguish. Fictional content occupies distinct regions.
- The finding is consistent across both probing and zero-shot behavioral experiments.

**Key insight**: LLMs conflate **distributional plausibility** with **epistemic justification**. Synthetic content that is linguistically plausible but epistemically ungrounded can override established beliefs, revealing that LLMs lack the epistemic robustness that would characterize genuine differentiation between justified (epistemic) and unjustified (non-epistemic) belief.

---

## 5. Theory of Mind and Epistemic Reasoning

### 5.1 MindGames: Formal Epistemic Logic (Sileo & Lernould, 2023)

**Paper:** "MindGames: Targeting Theory of Mind in Large Language Models with Dynamic Epistemic Modal Logic" (arXiv:2305.03353)

Created a procedurally generated benchmark using **Dynamic Epistemic Logic (DEL)** with ground truth verified by the SMCDEL model checker:

- 1,600 test problems across 4 observability setups (forehead-mud, thirst, etc.)
- **Zero-shot performance**: Near chance (~50%) for all model sizes tested (Pythia 70M-6.9B, GPT-3 family).
- **GPT-4**: 70% accuracy -- better than all others but far from human performance (94%).
- **No scaling trend**: Scaling model size alone does not produce robust epistemic reasoning.
- Human annotators' main error was conflating "know whether P" and "know that P" -- a subtle epistemic distinction that is central to the epistemic/non-epistemic boundary.

**Relevance**: LLMs largely fail at formal epistemic reasoning -- tracking what agents *know* (as opposed to what is merely *true*). This is a direct test of whether models can maintain the knowledge/belief distinction, and the results show they largely cannot.

### 5.2 Other ToM Benchmarks

Multiple other studies confirm related findings:

- **Evaluating LLMs for ToM** (Sap et al., 2023; arXiv:2302.02083): Systematic evaluation showing LLMs perform well on simple false belief tasks but fail on more complex or novel scenarios.
- **Hi-ToM** (arXiv:2310.16755): Higher-order Theory of Mind benchmark showing degrading performance with increasing belief nesting depth.
- **FANToM** (arXiv:2310.15421): Benchmark focusing on false beliefs in conversational contexts.
- **SimpleToM** (arXiv:2410.13648): Identifies a gap between mental state understanding and behavioral prediction in LLMs.
- **How Far Are LLMs from ToM Agents?** (arXiv:2310.03051): Comprehensive assessment finding that while LLMs show some ToM capabilities, they remain far from genuine belief reasoning agents.

---

## 6. Belief Revision and Calibration

### 6.1 Belief Revision in LLMs (arXiv:2309.02144)

Studies how LLMs update beliefs when presented with contradictory evidence. Key finding: LLMs show inconsistent belief revision patterns, sometimes maintaining original beliefs in the face of strong counter-evidence and other times abandoning well-supported beliefs too easily.

### 6.2 Calibration and Overconfidence (arXiv:1910.07514)

Research on reducing overconfidence in LLM predictions relates to the epistemic/non-epistemic distinction: a well-calibrated model would assign appropriate epistemic confidence to its beliefs, reflecting the strength of evidence rather than superficial features.

### 6.3 Evidence-to-Belief Bayesian Framework (arXiv:2504.19622)

Examines how LLMs process evidence to form beliefs, finding that while LLMs can approximate Bayesian updating in simple cases, they deviate significantly in more complex scenarios requiring integration of multiple evidence sources.

---

## 7. Synthesis: The Current State of Evidence

### 7.1 What LLMs Can Do

1. **Maintain separate representations** for self-beliefs vs. other-beliefs (Bortoletto et al., 2024).
2. **Track belief states** through specific computational mechanisms (lookback attention; Lanham et al., 2025).
3. **Update beliefs in response to evidence** through ICL in a roughly Bayesian manner (Bigelow et al., 2025).
4. **Perform simple false belief reasoning** in familiar ToM scenarios (various ToM papers).
5. **Encode epistemic familiarity** in activation space, distinguishing between content encountered in training and fabricated content (Dies et al., 2026).

### 7.2 What LLMs Cannot Do

1. **Distinguish knowledge from belief**: They fail to respect the factive nature of knowledge -- that "knowing P" entails P is true while "believing P" does not (Suzgun et al., 2024).
2. **Maintain epistemic stability**: Their belief-expression behavior is systematically modulated by non-epistemic prompt features (tone, role, intent) rather than being anchored in truth (Krastev et al., 2025).
3. **Resist distributional plausibility as a proxy for epistemic justification**: They conflate "linguistically plausible" with "epistemically grounded" (Dies et al., 2026).
4. **Perform formal epistemic reasoning**: They fail at modal logic tasks requiring tracking of knowledge states across agents and announcements (Sileo & Lernould, 2023).
5. **Handle first-person false beliefs**: They cannot adopt a believer's perspective when it contradicts their own knowledge (Suzgun et al., 2024).
6. **Satisfy uniformity criteria**: The same belief content receives different treatment depending on prompt framing (Herrmann & Levinstein, 2024; Krastev et al., 2025).

### 7.3 The Core Tension

The evidence reveals a **paradox**: at the representational level, LLMs appear to encode information relevant to epistemic distinctions (separate belief representations, epistemic familiarity, evidence vs. prior pathways). However, at the behavioral level, these internal distinctions do not reliably translate into robust epistemic reasoning. The models' outputs are systematically influenced by non-epistemic factors, suggesting that while the raw computational machinery for epistemic differentiation may exist, it is not sufficiently integrated or prioritized to produce genuinely epistemic behavior.

---

## 8. Key Experimental Directions for Further Investigation

Based on this review, the most promising experimental directions are:

1. **Cross-benchmark epistemic probing**: Use the KaBLE tasks to test whether internal representations (a la Bortoletto et al.) differentiate between knowledge and belief attributions, not just self vs. other beliefs.

2. **Epistemic fragility under controlled perturbation**: Extend the P-StaT framework to test whether epistemic perturbations (new evidence) and non-epistemic perturbations (tone/framing changes) produce distinguishable effects on internal representations.

3. **Bayesian decomposition of belief types**: Apply the Bigelow et al. framework to tasks that explicitly require epistemic vs. non-epistemic belief distinction, testing whether the evidence (ICL) and prior (steering) pathways map cleanly onto these categories.

4. **Lookback mechanism analysis for belief type**: Investigate whether the lookback mechanism (Lanham et al.) behaves differently for epistemic beliefs (beliefs formed from observation) vs. non-epistemic beliefs (beliefs based on values, desires, or social convention).

5. **Factive verb processing**: Design experiments using factive ("knows that") vs. non-factive ("believes that") verbs to test whether LLMs encode the presuppositional difference at the representation level.

---

## 9. Paper-by-Paper Summary Table

| # | arXiv ID | Short Title | Key Contribution | Datasets/Resources |
|---|----------|-------------|------------------|--------------------|
| 1 | 2410.21195 | KaBLE Benchmark | 13K questions testing belief/knowledge distinction; first-person asymmetry | KaBLE dataset, GitHub repo |
| 2 | 2405.21030 | Standards for Belief | Four criteria for genuine belief representations | Theoretical framework |
| 3 | 2402.18496 | RepBelief | Separate internal representations for self vs. other beliefs | BigToM dataset, probing code |
| 4 | 2505.14685 | Lookbacks Track Beliefs | Discovered lookback attention mechanism for belief tracking | CausalToM dataset |
| 5 | 2511.00617 | Belief Dynamics ICL | Bayesian framework unifying ICL (evidence) and steering (prior) | Anthropic persona evals |
| 6 | 2511.22746 | Epistemic Fragility | Prompt framing systematically modulates misinformation correction | 320 misinformation prompts |
| 7 | 2511.19166 | Representational Stability | Epistemic familiarity governs belief stability; P-StaT framework | Trilemma-of-Truth, Fictional datasets |
| 8 | 2305.03353 | MindGames | DEL-based epistemic reasoning benchmark; LLMs near chance | MindGames dataset (HF) |
| 9 | 2302.02083 | Evaluating LLMs ToM | Systematic ToM evaluation showing fragile performance | Various ToM benchmarks |
| 10 | 2310.16755 | Hi-ToM | Higher-order ToM; performance degrades with depth | Hi-ToM benchmark |
| 11 | 2310.15421 | FANToM | False belief in conversation | FANToM benchmark |
| 12 | 2410.13648 | SimpleToM | Gap between mental state understanding and behavioral prediction | SimpleToM dataset |
| 13 | 2310.03051 | How Far from ToM | Comprehensive ToM assessment | Multiple ToM benchmarks |
| 14 | 2501.15355 | Counterfactual Reflection | LLMs ToM via counterfactual reasoning | - |
| 15 | 2408.12022 | Epistemic Language Bayesian | Bayesian approach to epistemic language understanding | - |
| 16 | 2306.00924 | Minding LM Lack of ToM | Documents systematic ToM failures | - |
| 17 | 2407.06004 | Perceptions to Beliefs | From perceptual grounding to belief formation in LLMs | - |
| 18 | 2109.14723 | BeliefBank | Structured belief storage and consistency | BeliefBank dataset |
| 19 | 2502.06470 | Think Twice Perspective | Perspective-taking in ToM | - |
| 20 | 2410.16270 | Reflection Bench | Epistemic reflection benchmark | - |
| 21 | 2501.05032 | Fact Fiction Forecast | Representations of factual vs. fictional content | - |
| 22 | 1603.07704 | Probabilistic Coherence | Bayesian coherence in language models | - |
| 23 | 2504.19622 | Evidence to Belief | Bayesian evidence integration | - |
| 24 | 1910.07514 | Reducing Overconfidence | Calibration and epistemic confidence | - |
| 25 | 2510.09033 | LLMs Know What Humans Know | Modeling human knowledge attribution | - |
| 26 | 2407.15814 | Linguistic Uncertainty | Perceptions of uncertainty expressions | - |
| 27 | 2309.02144 | Belief Revision | How LLMs update beliefs with contradictory evidence | - |
| 28 | 2408.07237 | Semantic Embedding Beliefs | Embedding-level belief representations | - |
| 29 | 2505.14685 | Lookbacks Track Beliefs | Mechanistic belief tracking via attention | CausalToM dataset |

---

## 10. Conclusion

The current evidence suggests that LLMs have **partial but incomplete** differentiation between epistemic and non-epistemic belief. Internally, they maintain representations that encode aspects relevant to epistemic distinctions (familiarity, observational access, evidence vs. priors). Behaviorally, however, they fail to consistently deploy these distinctions: they confuse knowledge with belief, allow non-epistemic contextual features to override epistemic commitments, and cannot robustly perform formal epistemic reasoning. The gap between representational capacity and behavioral competence represents both a key finding and a primary target for future research.
