"""
Main experiment runner: Do LLMs differentiate epistemic belief from non-epistemic belief?

Runs 4 experiments using OpenAI API (GPT-4.1):
1. Explicit belief classification
2. Differential response elicitation
3. Belief revision under counterevidence
4. Factive verb sensitivity
"""

import json
import os
import random
import time
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from openai import OpenAI

from stimuli import (
    EPISTEMIC_BELIEFS,
    NON_EPISTEMIC_BELIEFS,
    VERB_SENSITIVITY_STATEMENTS,
    COUNTEREVIDENCE,
)

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Configuration
MODEL = "gpt-4.1"
TEMPERATURE = 0.3  # Low but non-zero for slight variation across runs
MAX_TOKENS = 500
NUM_RUNS = 3  # Repeat each experiment for consistency

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def call_llm(prompt: str, system_prompt: str = "", temperature: float = TEMPERATURE) -> str:
    """Call the LLM with retry logic."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=MAX_TOKENS,
                seed=SEED,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < 4:
                wait = 2 ** attempt
                print(f"  API error (attempt {attempt+1}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  API failed after 5 attempts: {e}")
                return f"ERROR: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 1: Explicit Belief Classification
# ─────────────────────────────────────────────────────────────────────────────
def run_experiment_1():
    """Test whether LLMs can explicitly classify beliefs as epistemic vs. non-epistemic."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Explicit Belief Classification")
    print("=" * 70)

    system_prompt = (
        "You are a philosophy assistant. When given a belief statement, classify it as either "
        "'epistemic' (a belief grounded in evidence, facts, or empirical observation that can be "
        "verified as true or false) or 'non-epistemic' (a belief grounded in values, preferences, "
        "faith, or normative judgments that cannot be straightforwardly verified as true or false). "
        "Respond with ONLY the word 'epistemic' or 'non_epistemic', nothing else."
    )

    all_beliefs = [(b, "epistemic") for b in EPISTEMIC_BELIEFS] + [
        (b, "non_epistemic") for b in NON_EPISTEMIC_BELIEFS
    ]
    random.shuffle(all_beliefs)

    results = []
    for run_idx in range(NUM_RUNS):
        print(f"\n--- Run {run_idx + 1}/{NUM_RUNS} ---")
        run_results = []
        for belief, true_label in all_beliefs:
            prompt = f'Classify this belief: "I believe that {belief}."'
            response = call_llm(prompt, system_prompt)
            predicted = "epistemic" if "epistemic" in response.lower() and "non" not in response.lower() else "non_epistemic"
            correct = predicted == true_label
            run_results.append({
                "belief": belief,
                "true_label": true_label,
                "predicted": predicted,
                "raw_response": response,
                "correct": correct,
                "run": run_idx,
            })
            status = "OK" if correct else "WRONG"
            print(f"  [{status}] {true_label:14s} -> {predicted:14s} | {belief[:60]}")
        results.extend(run_results)

    # Save results
    with open(RESULTS_DIR / "exp1_classification.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    correct_count = sum(r["correct"] for r in results)
    total = len(results)
    print(f"\nExp 1 Overall Accuracy: {correct_count}/{total} = {correct_count/total:.3f}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 2: Differential Response Elicitation
# ─────────────────────────────────────────────────────────────────────────────
def run_experiment_2():
    """Test whether LLMs spontaneously respond differently to epistemic vs. non-epistemic beliefs."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Differential Response Elicitation")
    print("=" * 70)

    # Coding prompt for automated response analysis
    coding_system = (
        "You are a response analyzer. Given a response to a belief statement, code it on these dimensions. "
        "Respond ONLY with a JSON object (no markdown, no explanation):\n"
        '{"challenges": true/false (does the response challenge or question the belief?), '
        '"uses_evidence": true/false (does the response cite evidence, data, or facts?), '
        '"uses_values": true/false (does the response appeal to values, ethics, or personal judgment?), '
        '"hedges": true/false (does the response use hedging language like "it depends", "some might argue"?), '
        '"agreement_level": 1-5 (1=strongly disagrees, 3=neutral, 5=strongly agrees)}'
    )

    all_beliefs = [(b, "epistemic") for b in EPISTEMIC_BELIEFS] + [
        (b, "non_epistemic") for b in NON_EPISTEMIC_BELIEFS
    ]

    results = []
    for run_idx in range(NUM_RUNS):
        print(f"\n--- Run {run_idx + 1}/{NUM_RUNS} ---")
        for belief, belief_type in all_beliefs:
            # Get model's natural response
            prompt = f'A person says to you: "I believe that {belief}." How would you respond to this person?'
            response = call_llm(prompt)

            # Automated coding of the response
            coding_prompt = (
                f'The original belief statement was: "I believe that {belief}."\n'
                f"The response was:\n{response}\n\n"
                "Code this response as a JSON object."
            )
            coding_raw = call_llm(coding_prompt, coding_system, temperature=0.0)

            try:
                # Parse JSON - handle potential markdown wrapping
                coding_text = coding_raw
                if "```" in coding_text:
                    coding_text = coding_text.split("```")[1]
                    if coding_text.startswith("json"):
                        coding_text = coding_text[4:]
                coding = json.loads(coding_text.strip())
            except (json.JSONDecodeError, IndexError):
                coding = {
                    "challenges": None,
                    "uses_evidence": None,
                    "uses_values": None,
                    "hedges": None,
                    "agreement_level": None,
                    "parse_error": coding_raw,
                }

            result = {
                "belief": belief,
                "belief_type": belief_type,
                "response": response,
                "coding": coding,
                "run": run_idx,
            }
            results.append(result)
            print(f"  {belief_type:14s} | challenges={coding.get('challenges', '?'):>5} | evidence={coding.get('uses_evidence', '?'):>5} | values={coding.get('uses_values', '?'):>5} | agree={coding.get('agreement_level', '?')} | {belief[:40]}")

    with open(RESULTS_DIR / "exp2_differential_response.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 3: Belief Revision Under Counterevidence
# ─────────────────────────────────────────────────────────────────────────────
def run_experiment_3():
    """Test whether LLMs recommend different revision for epistemic vs. non-epistemic beliefs."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Belief Revision Under Counterevidence")
    print("=" * 70)

    coding_system = (
        "You are a response analyzer. Given a response about whether someone should revise their belief, "
        "code it. Respond ONLY with a JSON object (no markdown):\n"
        '{"recommends_revision": true/false (does the response recommend changing the belief?), '
        '"revision_strength": 1-5 (1=strongly maintain belief, 3=uncertain, 5=strongly revise), '
        '"cites_evidence_quality": true/false (does it evaluate the quality of the counterevidence?), '
        '"acknowledges_subjectivity": true/false (does it note the belief is subjective/value-based?), '
        '"nuanced": true/false (does it offer a nuanced rather than binary response?)}'
    )

    all_beliefs = [(b, "epistemic") for b in EPISTEMIC_BELIEFS] + [
        (b, "non_epistemic") for b in NON_EPISTEMIC_BELIEFS
    ]

    results = []
    for run_idx in range(NUM_RUNS):
        print(f"\n--- Run {run_idx + 1}/{NUM_RUNS} ---")
        for belief, belief_type in all_beliefs:
            counter = COUNTEREVIDENCE.get(belief, "Some people disagree with this view.")
            prompt = (
                f'Alex firmly believes that {belief}. '
                f'However, Alex has just encountered the following information: "{counter}" '
                f'Should Alex revise their belief? Why or why not?'
            )
            response = call_llm(prompt)

            coding_prompt = (
                f'The original belief was: "{belief}" (type: {belief_type})\n'
                f'The counterevidence was: "{counter}"\n'
                f"The response about revision was:\n{response}\n\n"
                "Code this response as a JSON object."
            )
            coding_raw = call_llm(coding_prompt, coding_system, temperature=0.0)

            try:
                coding_text = coding_raw
                if "```" in coding_text:
                    coding_text = coding_text.split("```")[1]
                    if coding_text.startswith("json"):
                        coding_text = coding_text[4:]
                coding = json.loads(coding_text.strip())
            except (json.JSONDecodeError, IndexError):
                coding = {
                    "recommends_revision": None,
                    "revision_strength": None,
                    "cites_evidence_quality": None,
                    "acknowledges_subjectivity": None,
                    "nuanced": None,
                    "parse_error": coding_raw,
                }

            result = {
                "belief": belief,
                "belief_type": belief_type,
                "counterevidence": counter,
                "response": response,
                "coding": coding,
                "run": run_idx,
            }
            results.append(result)
            rev = coding.get("revision_strength", "?")
            print(f"  {belief_type:14s} | revision_strength={rev} | {belief[:50]}")

    with open(RESULTS_DIR / "exp3_belief_revision.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 4: Factive Verb Sensitivity
# ─────────────────────────────────────────────────────────────────────────────
def run_experiment_4():
    """Test whether substituting 'knows', 'believes', and 'values' changes model behavior."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Factive Verb Sensitivity")
    print("=" * 70)

    verbs = ["knows", "believes", "values"]

    coding_system = (
        "You are a response analyzer. Given a response about a person's stance toward a proposition, "
        "code it. Respond ONLY with a JSON object (no markdown):\n"
        '{"implies_truth": true/false (does the response treat the proposition as factually true?), '
        '"revisable": true/false (does the response suggest the person could/should change their stance?), '
        '"treats_as_factual": true/false (is the proposition treated as a factual claim?), '
        '"treats_as_value": true/false (is the proposition treated as a value judgment?), '
        '"certainty_level": 1-5 (1=very uncertain, 5=very certain about the proposition)}'
    )

    results = []
    for run_idx in range(NUM_RUNS):
        print(f"\n--- Run {run_idx + 1}/{NUM_RUNS} ---")
        for statement, stmt_type in VERB_SENSITIVITY_STATEMENTS:
            for verb in verbs:
                prompt = (
                    f'Alex {verb} {statement}. '
                    f'Based on this information alone: '
                    f'1) What can we conclude about whether "{statement}" is true? '
                    f'2) Should Alex change their mind about this? '
                    f'Please answer both questions.'
                )
                response = call_llm(prompt)

                coding_prompt = (
                    f'The statement was: "Alex {verb} {statement}."\n'
                    f"The response was:\n{response}\n\n"
                    "Code this response as a JSON object."
                )
                coding_raw = call_llm(coding_prompt, coding_system, temperature=0.0)

                try:
                    coding_text = coding_raw
                    if "```" in coding_text:
                        coding_text = coding_text.split("```")[1]
                        if coding_text.startswith("json"):
                            coding_text = coding_text[4:]
                    coding = json.loads(coding_text.strip())
                except (json.JSONDecodeError, IndexError):
                    coding = {
                        "implies_truth": None,
                        "revisable": None,
                        "treats_as_factual": None,
                        "treats_as_value": None,
                        "certainty_level": None,
                        "parse_error": coding_raw,
                    }

                result = {
                    "statement": statement,
                    "statement_type": stmt_type,
                    "verb": verb,
                    "response": response,
                    "coding": coding,
                    "run": run_idx,
                }
                results.append(result)
                truth = coding.get("implies_truth", "?")
                cert = coding.get("certainty_level", "?")
                print(f"  {verb:8s} | truth={truth!s:>5} | certainty={cert} | {statement[:40]}")

    with open(RESULTS_DIR / "exp4_factive_verb.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Starting experiments at {datetime.now().isoformat()}")
    print(f"Model: {MODEL}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Seed: {SEED}")
    print(f"Runs per experiment: {NUM_RUNS}")
    print(f"Python: {sys.version}")

    # Save configuration
    config = {
        "model": MODEL,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "seed": SEED,
        "num_runs": NUM_RUNS,
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "n_epistemic_beliefs": len(EPISTEMIC_BELIEFS),
        "n_non_epistemic_beliefs": len(NON_EPISTEMIC_BELIEFS),
        "n_verb_statements": len(VERB_SENSITIVITY_STATEMENTS),
    }
    with open(RESULTS_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Run all experiments
    exp1_results = run_experiment_1()
    exp2_results = run_experiment_2()
    exp3_results = run_experiment_3()
    exp4_results = run_experiment_4()

    print(f"\nAll experiments completed at {datetime.now().isoformat()}")
    print(f"Results saved to {RESULTS_DIR}/")
