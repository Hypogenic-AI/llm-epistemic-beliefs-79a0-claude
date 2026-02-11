# Research Plan: Do LLMs Differentiate Epistemic Belief from Non-Epistemic Belief?

## Motivation & Novelty Assessment

### Why This Research Matters
Vesga et al. argue that humans maintain categorically different types of beliefs — epistemic beliefs grounded in evidence and truth-tracking ("I believe the Earth is round because of satellite imagery") vs. non-epistemic beliefs anchored in values, desires, or faith ("I believe in treating people fairly"). This distinction is fundamental to human cognition: we reason differently about these belief types, apply different revision standards, and evaluate them by different criteria. If LLMs similarly differentiate belief types, this has profound implications for AI safety (should models challenge value-beliefs?), alignment (how should models engage with beliefs they "disagree" with?), and our understanding of what LLMs represent.

### Gap in Existing Work
The literature review reveals extensive work on *epistemic reasoning about beliefs* (KaBLE, MindGames, ToM benchmarks) but a critical gap: **no study directly tests whether LLMs behaviorally differentiate between epistemic and non-epistemic beliefs as categories**. Prior work asks "can LLMs reason about what someone knows vs. believes?" but not "do LLMs treat evidence-based beliefs differently from value-based beliefs?" The closest work — Bigelow et al.'s Bayesian decomposition showing ICL (evidence) and steering (prior) as separate pathways — operates at the mechanistic level but has not been tested behaviorally with explicit belief type contrasts.

### Our Novel Contribution
We design and run a behavioral experiment that directly tests whether frontier LLMs (GPT-4.1) differentiate their responses to epistemic vs. non-epistemic beliefs across four dimensions:
1. **Correctability** — Are models more willing to challenge epistemic beliefs than non-epistemic ones?
2. **Justification type** — Do models offer evidence for epistemic beliefs but appeals to values/authority for non-epistemic ones?
3. **Revision behavior** — When presented with counterevidence, do models revise epistemic beliefs more readily?
4. **Factive sensitivity** — Do models treat "know," "believe," and "value" differently when applied to the same content?

### Experiment Justification
- **Experiment 1 (Belief Classification)**: Tests whether LLMs can explicitly categorize beliefs into epistemic vs. non-epistemic when asked directly. Baseline capability check.
- **Experiment 2 (Differential Response)**: Tests whether LLMs *spontaneously* treat the two belief types differently in their reasoning, justification, and willingness to challenge — without being told about the distinction.
- **Experiment 3 (Revision Asymmetry)**: Tests whether counterevidence affects epistemic and non-epistemic beliefs differently, as it does for humans (we revise factual beliefs upon evidence but hold moral beliefs more firmly).
- **Experiment 4 (Factive Verb Sensitivity)**: Tests whether substituting "knows," "believes," and "values" before the same content changes model behavior — a direct test of whether the model encodes the epistemic/non-epistemic distinction at the linguistic level.

---

## Research Question
Do frontier LLMs behaviorally differentiate between epistemic beliefs (evidence-based, truth-tracking) and non-epistemic beliefs (value-based, desire-based, faith-based), and if so, in what ways?

## Background and Motivation
Inspired by Vesga et al.'s argument that humans maintain different types of beliefs, we test whether LLMs exhibit analogous differentiation. The literature shows LLMs struggle with epistemic reasoning (KaBLE: 54.4% on false belief tasks) and are susceptible to non-epistemic prompt manipulation (Krastev et al.: creative framing reduces correction by 89%). However, no study directly contrasts how LLMs treat beliefs of different epistemic types.

## Hypothesis Decomposition

**H1**: LLMs can explicitly classify beliefs as epistemic vs. non-epistemic when asked.
- IV: Belief type (epistemic/non-epistemic). DV: Classification accuracy.

**H2**: LLMs spontaneously produce different response patterns for epistemic vs. non-epistemic beliefs.
- IV: Belief type. DV: Response features (evidence-citing, willingness to challenge, hedging behavior).

**H3**: LLMs show asymmetric revision behavior — more willing to revise epistemic beliefs given counterevidence than non-epistemic beliefs.
- IV: Belief type × counterevidence presence. DV: Revision rate.

**H4**: LLMs are sensitive to factive verbs — treating "X knows P," "X believes P," and "X values P" differently even when P is held constant.
- IV: Verb type (knows/believes/values). DV: Response behavior.

## Proposed Methodology

### Approach
We use a **behavioral probing** approach: present frontier LLMs with carefully constructed prompts containing beliefs of different types and analyze their responses. This is a black-box approach using API calls — appropriate since we want to test whether the distinction manifests in observable behavior, not just internal representations.

### Stimulus Construction
We create a balanced set of 40 belief statements:
- 20 epistemic beliefs (factual claims with evidence basis): "The Earth orbits the Sun," "Water boils at 100°C at sea level," etc.
- 20 non-epistemic beliefs (values, preferences, faith): "Honesty is the best policy," "Everyone deserves equal rights," "Hard work leads to success," etc.
Each belief is presented in matched pairs controlling for complexity and familiarity.

### Experimental Steps

1. **Experiment 1 — Explicit Classification** (Baseline)
   - Present LLM with 40 beliefs and ask: "Is this an epistemic belief (based on evidence/facts) or a non-epistemic belief (based on values/desires/faith)?"
   - Measure classification accuracy against our ground truth labels.
   - This establishes whether the model can even distinguish the categories.

2. **Experiment 2 — Differential Response Elicitation**
   - For each belief, prompt: "A person says: 'I believe [X].' How would you respond to this person?"
   - Code responses for: (a) agreement/challenge, (b) evidence-based vs. value-based reasoning, (c) hedging language, (d) request for clarification.
   - Use automated coding via a separate LLM call + manual verification on subset.

3. **Experiment 3 — Belief Revision Under Counterevidence**
   - For each belief, present: "[Person] believes [X]. New information suggests [not-X]. How should [Person] update their belief?"
   - Measure: recommendation to revise (yes/no/partial), strength of revision recommendation, type of reasoning offered.
   - Prediction: LLMs should recommend stronger revision for epistemic beliefs.

4. **Experiment 4 — Factive Verb Sensitivity**
   - Take 10 statements usable in both epistemic and non-epistemic frames.
   - Present with three verbs: "Alex knows that [P]," "Alex believes that [P]," "Alex values [P]."
   - Ask: "What can we conclude about [P]?" and "Should Alex change their mind about [P]?"
   - Measure whether verb choice affects conclusions about truth status and revisability.

### Baselines
- **Random baseline**: 50% accuracy on classification.
- **Prior work reference**: KaBLE results (85.7% factual, 54.4% false belief) as context.
- **Within-experiment**: Compare epistemic vs. non-epistemic response distributions as main contrast.

### Evaluation Metrics
- Experiment 1: Classification accuracy, Cohen's kappa for agreement with ground truth.
- Experiment 2: Chi-square test on response category distributions (challenge vs. agree, evidence vs. values).
- Experiment 3: Paired t-test on revision scores (epistemic vs. non-epistemic).
- Experiment 4: Repeated-measures ANOVA on response patterns across verb conditions.

### Statistical Analysis Plan
- Significance level: α = 0.05
- Effect sizes: Cohen's d for continuous comparisons, Cramér's V for categorical.
- Multiple comparison correction: Bonferroni where applicable.
- 95% confidence intervals reported for all main effects.
- 3 independent runs per condition to assess consistency (temperature > 0).

## Expected Outcomes
- **Supporting hypothesis**: LLMs classify belief types accurately (>80%), show differential response patterns, recommend stronger revision for epistemic beliefs, and are sensitive to factive verbs.
- **Refuting hypothesis**: LLMs treat all belief types identically (no significant difference across conditions).
- **Partial support (most likely)**: LLMs show some differentiation but with interesting failures — e.g., they may classify correctly but fail to revise their own behavior, or show verb sensitivity without belief-type differentiation.

## Timeline and Milestones
1. Environment setup + stimulus creation: 15 min
2. Experiment 1 implementation + run: 20 min
3. Experiment 2 implementation + run: 25 min
4. Experiment 3 implementation + run: 25 min
5. Experiment 4 implementation + run: 20 min
6. Analysis + visualization: 30 min
7. Documentation: 25 min

## Potential Challenges
- **API rate limits**: Mitigate with retry logic and caching.
- **Response parsing**: LLM outputs may not follow expected formats. Use flexible parsing.
- **Subjectivity in coding**: Use automated coding with validation.
- **Belief classification ambiguity**: Some beliefs straddle epistemic/non-epistemic line. Use clear prototypical examples.

## Success Criteria
1. All experiments produce usable data from real LLM API calls.
2. Statistical tests reveal whether differences between belief types are significant.
3. Results are documented with full reproducibility information.
4. Findings contribute a novel data point to the literature on LLM belief differentiation.
