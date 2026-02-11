"""
Analysis and visualization for epistemic vs. non-epistemic belief differentiation experiments.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from scipy import stats

RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.2)
plt.rcParams["figure.dpi"] = 150


def load_results(filename):
    with open(RESULTS_DIR / filename) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 1 ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def analyze_exp1():
    print("\n" + "=" * 70)
    print("ANALYSIS: Experiment 1 — Explicit Belief Classification")
    print("=" * 70)

    data = load_results("exp1_classification.json")
    df = pd.DataFrame(data)

    # Overall accuracy
    overall_acc = df["correct"].mean()
    print(f"Overall accuracy: {overall_acc:.3f} ({df['correct'].sum()}/{len(df)})")

    # Accuracy by belief type
    for bt in ["epistemic", "non_epistemic"]:
        subset = df[df["true_label"] == bt]
        acc = subset["correct"].mean()
        n = len(subset)
        # Wilson CI
        z = 1.96
        p_hat = acc
        denom = 1 + z**2 / n
        center = (p_hat + z**2 / (2 * n)) / denom
        margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
        print(f"  {bt:14s}: {acc:.3f} (n={n}) [95% CI: {center-margin:.3f}, {center+margin:.3f}]")

    # Accuracy by run
    for run in sorted(df["run"].unique()):
        run_acc = df[df["run"] == run]["correct"].mean()
        print(f"  Run {run}: {run_acc:.3f}")

    # Confusion matrix
    from collections import Counter
    confusion = Counter()
    for _, row in df.iterrows():
        confusion[(row["true_label"], row["predicted"])] += 1

    print("\nConfusion Matrix:")
    print(f"                  Predicted")
    print(f"                  epistemic  non_epistemic")
    print(f"  True epistemic  {confusion[('epistemic', 'epistemic')]:>5d}      {confusion[('epistemic', 'non_epistemic')]:>5d}")
    print(f"  True non_epist  {confusion[('non_epistemic', 'epistemic')]:>5d}      {confusion[('non_epistemic', 'non_epistemic')]:>5d}")

    # Cohen's kappa
    n_total = len(df)
    po = overall_acc
    pe_ep = (confusion[("epistemic", "epistemic")] + confusion[("epistemic", "non_epistemic")]) / n_total * \
            (confusion[("epistemic", "epistemic")] + confusion[("non_epistemic", "epistemic")]) / n_total
    pe_ne = (confusion[("non_epistemic", "epistemic")] + confusion[("non_epistemic", "non_epistemic")]) / n_total * \
            (confusion[("epistemic", "non_epistemic")] + confusion[("non_epistemic", "non_epistemic")]) / n_total
    pe = pe_ep + pe_ne
    kappa = (po - pe) / (1 - pe) if pe < 1 else 0
    print(f"\nCohen's Kappa: {kappa:.3f}")

    # Plot: accuracy by belief type per run
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart of accuracy by type
    acc_by_type = df.groupby("true_label")["correct"].mean()
    ci_by_type = df.groupby("true_label")["correct"].sem() * 1.96
    ax = axes[0]
    bars = acc_by_type.plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452"], yerr=ci_by_type, capsize=5)
    ax.set_title("Classification Accuracy by Belief Type")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("True Belief Type")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
    ax.legend()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    # Confusion matrix heatmap
    ax = axes[1]
    cm = np.array([
        [confusion[("epistemic", "epistemic")], confusion[("epistemic", "non_epistemic")]],
        [confusion[("non_epistemic", "epistemic")], confusion[("non_epistemic", "non_epistemic")]],
    ])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["epistemic", "non_epistemic"],
                yticklabels=["epistemic", "non_epistemic"])
    ax.set_title("Confusion Matrix (all runs)")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "exp1_classification.png", bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {PLOTS_DIR / 'exp1_classification.png'}")

    return {
        "overall_accuracy": overall_acc,
        "epistemic_accuracy": df[df["true_label"] == "epistemic"]["correct"].mean(),
        "non_epistemic_accuracy": df[df["true_label"] == "non_epistemic"]["correct"].mean(),
        "cohens_kappa": kappa,
        "n_total": n_total,
    }


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 2 ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def analyze_exp2():
    print("\n" + "=" * 70)
    print("ANALYSIS: Experiment 2 — Differential Response Elicitation")
    print("=" * 70)

    data = load_results("exp2_differential_response.json")

    # Extract coded features
    rows = []
    for d in data:
        coding = d["coding"]
        if coding.get("agreement_level") is not None:
            rows.append({
                "belief": d["belief"],
                "belief_type": d["belief_type"],
                "run": d["run"],
                "challenges": bool(coding.get("challenges", False)),
                "uses_evidence": bool(coding.get("uses_evidence", False)),
                "uses_values": bool(coding.get("uses_values", False)),
                "hedges": bool(coding.get("hedges", False)),
                "agreement_level": int(coding["agreement_level"]),
            })

    df = pd.DataFrame(rows)
    print(f"Valid coded responses: {len(df)}")

    # Mean features by belief type
    features = ["challenges", "uses_evidence", "uses_values", "hedges", "agreement_level"]
    print("\nMean values by belief type:")
    for feat in features:
        ep = df[df["belief_type"] == "epistemic"][feat]
        ne = df[df["belief_type"] == "non_epistemic"][feat]
        print(f"  {feat:20s}: epistemic={ep.mean():.3f} (sd={ep.std():.3f}), non_epistemic={ne.mean():.3f} (sd={ne.std():.3f})")

        # Statistical test
        if feat == "agreement_level":
            stat, p = stats.mannwhitneyu(ep, ne, alternative="two-sided")
            d_cohens = (ep.mean() - ne.mean()) / np.sqrt((ep.std()**2 + ne.std()**2) / 2) if ep.std() + ne.std() > 0 else 0
            print(f"    Mann-Whitney U={stat:.1f}, p={p:.4f}, Cohen's d={d_cohens:.3f}")
        else:
            # Chi-square for binary features
            ct = pd.crosstab(df["belief_type"], df[feat])
            if ct.shape == (2, 2):
                chi2, p, dof, expected = stats.chi2_contingency(ct)
                cramers_v = np.sqrt(chi2 / len(df))
                print(f"    Chi-square={chi2:.2f}, p={p:.4f}, Cramér's V={cramers_v:.3f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Binary features comparison
    ax = axes[0]
    binary_feats = ["challenges", "uses_evidence", "uses_values", "hedges"]
    means = df.groupby("belief_type")[binary_feats].mean()
    means.T.plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452"])
    ax.set_title("Response Features by Belief Type")
    ax.set_ylabel("Proportion")
    ax.set_xlabel("Feature")
    ax.legend(title="Belief Type")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")

    # Agreement level distribution
    ax = axes[1]
    for bt, color in [("epistemic", "#4C72B0"), ("non_epistemic", "#DD8452")]:
        subset = df[df["belief_type"] == bt]["agreement_level"]
        ax.hist(subset, bins=np.arange(0.5, 6.5, 1), alpha=0.6, color=color, label=bt, edgecolor="black")
    ax.set_title("Agreement Level Distribution")
    ax.set_xlabel("Agreement Level (1=disagree, 5=agree)")
    ax.set_ylabel("Count")
    ax.legend()

    # Agreement level box plot
    ax = axes[2]
    df.boxplot(column="agreement_level", by="belief_type", ax=ax)
    ax.set_title("Agreement Level by Belief Type")
    ax.set_xlabel("Belief Type")
    ax.set_ylabel("Agreement Level")
    plt.suptitle("")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "exp2_differential_response.png", bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {PLOTS_DIR / 'exp2_differential_response.png'}")

    return {
        "n_valid": len(df),
        "means_by_type": df.groupby("belief_type")[features].mean().to_dict(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 3 ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def analyze_exp3():
    print("\n" + "=" * 70)
    print("ANALYSIS: Experiment 3 — Belief Revision Under Counterevidence")
    print("=" * 70)

    data = load_results("exp3_belief_revision.json")

    rows = []
    for d in data:
        coding = d["coding"]
        if coding.get("revision_strength") is not None:
            rows.append({
                "belief": d["belief"],
                "belief_type": d["belief_type"],
                "run": d["run"],
                "recommends_revision": bool(coding.get("recommends_revision", False)),
                "revision_strength": int(coding["revision_strength"]),
                "cites_evidence_quality": bool(coding.get("cites_evidence_quality", False)),
                "acknowledges_subjectivity": bool(coding.get("acknowledges_subjectivity", False)),
                "nuanced": bool(coding.get("nuanced", False)),
            })

    df = pd.DataFrame(rows)
    print(f"Valid coded responses: {len(df)}")

    # Revision strength by belief type
    ep = df[df["belief_type"] == "epistemic"]["revision_strength"]
    ne = df[df["belief_type"] == "non_epistemic"]["revision_strength"]
    print(f"\nRevision strength:")
    print(f"  Epistemic:     mean={ep.mean():.3f} (sd={ep.std():.3f})")
    print(f"  Non-epistemic: mean={ne.mean():.3f} (sd={ne.std():.3f})")

    # Statistical test
    stat, p = stats.mannwhitneyu(ep, ne, alternative="two-sided")
    d_cohens = (ep.mean() - ne.mean()) / np.sqrt((ep.std()**2 + ne.std()**2) / 2) if ep.std() + ne.std() > 0 else 0
    print(f"  Mann-Whitney U={stat:.1f}, p={p:.6f}, Cohen's d={d_cohens:.3f}")

    # Revision recommendation rate
    for bt in ["epistemic", "non_epistemic"]:
        subset = df[df["belief_type"] == bt]
        rate = subset["recommends_revision"].mean()
        print(f"  Revision recommendation rate ({bt}): {rate:.3f}")

    # Chi-square on recommends_revision
    ct = pd.crosstab(df["belief_type"], df["recommends_revision"])
    if ct.shape == (2, 2):
        chi2, p_chi, dof, _ = stats.chi2_contingency(ct)
        print(f"  Chi-square on revision recommendation: chi2={chi2:.2f}, p={p_chi:.4f}")

    # Evidence quality and subjectivity differences
    for feat in ["cites_evidence_quality", "acknowledges_subjectivity", "nuanced"]:
        for bt in ["epistemic", "non_epistemic"]:
            rate = df[df["belief_type"] == bt][feat].mean()
            print(f"  {feat} ({bt}): {rate:.3f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Revision strength distributions
    ax = axes[0]
    for bt, color in [("epistemic", "#4C72B0"), ("non_epistemic", "#DD8452")]:
        subset = df[df["belief_type"] == bt]["revision_strength"]
        ax.hist(subset, bins=np.arange(0.5, 6.5, 1), alpha=0.6, color=color, label=bt, edgecolor="black")
    ax.set_title("Revision Strength Distribution")
    ax.set_xlabel("Revision Strength (1=maintain, 5=revise)")
    ax.set_ylabel("Count")
    ax.legend()

    # Revision strength comparison
    ax = axes[1]
    df.boxplot(column="revision_strength", by="belief_type", ax=ax)
    ax.set_title("Revision Strength by Belief Type")
    ax.set_xlabel("Belief Type")
    ax.set_ylabel("Revision Strength")
    plt.suptitle("")

    # Binary features
    ax = axes[2]
    binary_feats = ["recommends_revision", "cites_evidence_quality", "acknowledges_subjectivity", "nuanced"]
    means = df.groupby("belief_type")[binary_feats].mean()
    means.T.plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452"])
    ax.set_title("Response Features by Belief Type")
    ax.set_ylabel("Proportion")
    ax.legend(title="Belief Type")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "exp3_belief_revision.png", bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {PLOTS_DIR / 'exp3_belief_revision.png'}")

    return {
        "n_valid": len(df),
        "epistemic_revision_mean": ep.mean(),
        "non_epistemic_revision_mean": ne.mean(),
        "mann_whitney_p": p,
        "cohens_d": d_cohens,
    }


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 4 ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def analyze_exp4():
    print("\n" + "=" * 70)
    print("ANALYSIS: Experiment 4 — Factive Verb Sensitivity")
    print("=" * 70)

    data = load_results("exp4_factive_verb.json")

    rows = []
    for d in data:
        coding = d["coding"]
        if coding.get("certainty_level") is not None:
            rows.append({
                "statement": d["statement"],
                "statement_type": d["statement_type"],
                "verb": d["verb"],
                "run": d["run"],
                "implies_truth": bool(coding.get("implies_truth", False)),
                "revisable": bool(coding.get("revisable", False)),
                "treats_as_factual": bool(coding.get("treats_as_factual", False)),
                "treats_as_value": bool(coding.get("treats_as_value", False)),
                "certainty_level": int(coding["certainty_level"]),
            })

    df = pd.DataFrame(rows)
    print(f"Valid coded responses: {len(df)}")

    # Certainty by verb type
    print("\nCertainty level by verb:")
    for verb in ["knows", "believes", "values"]:
        subset = df[df["verb"] == verb]["certainty_level"]
        print(f"  {verb:10s}: mean={subset.mean():.3f} (sd={subset.std():.3f})")

    # Kruskal-Wallis test across verbs
    groups = [df[df["verb"] == v]["certainty_level"].values for v in ["knows", "believes", "values"]]
    h_stat, p_kw = stats.kruskal(*groups)
    print(f"  Kruskal-Wallis H={h_stat:.2f}, p={p_kw:.4f}")

    # Pairwise comparisons
    for v1, v2 in [("knows", "believes"), ("knows", "values"), ("believes", "values")]:
        g1 = df[df["verb"] == v1]["certainty_level"]
        g2 = df[df["verb"] == v2]["certainty_level"]
        stat, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
        p_adj = min(p * 3, 1.0)  # Bonferroni
        print(f"  {v1} vs {v2}: U={stat:.1f}, p={p:.4f}, p_adj={p_adj:.4f}")

    # Implies truth by verb
    print("\nImplies truth by verb:")
    for verb in ["knows", "believes", "values"]:
        rate = df[df["verb"] == verb]["implies_truth"].mean()
        print(f"  {verb:10s}: {rate:.3f}")

    # Revisable by verb
    print("\nRevisable by verb:")
    for verb in ["knows", "believes", "values"]:
        rate = df[df["verb"] == verb]["revisable"].mean()
        print(f"  {verb:10s}: {rate:.3f}")

    # Interaction: verb × statement_type
    print("\nCertainty by verb × statement type:")
    for stype in df["statement_type"].unique():
        print(f"  Statement type: {stype}")
        for verb in ["knows", "believes", "values"]:
            subset = df[(df["verb"] == verb) & (df["statement_type"] == stype)]["certainty_level"]
            if len(subset) > 0:
                print(f"    {verb:10s}: mean={subset.mean():.3f} (n={len(subset)})")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Certainty by verb
    ax = axes[0]
    verb_order = ["knows", "believes", "values"]
    sns.boxplot(data=df, x="verb", y="certainty_level", order=verb_order, ax=ax,
                palette=["#4C72B0", "#55A868", "#DD8452"])
    ax.set_title("Certainty Level by Verb Type")
    ax.set_xlabel("Verb")
    ax.set_ylabel("Certainty Level")

    # Implies truth by verb
    ax = axes[1]
    truth_rates = df.groupby("verb")["implies_truth"].mean().reindex(verb_order)
    truth_rates.plot(kind="bar", ax=ax, color=["#4C72B0", "#55A868", "#DD8452"])
    ax.set_title("Implies Truth Rate by Verb")
    ax.set_ylabel("Proportion implying truth")
    ax.set_xlabel("Verb")
    ax.set_ylim(0, 1.1)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    # Heatmap: certainty by verb × statement type
    ax = axes[2]
    pivot = df.pivot_table(values="certainty_level", index="statement_type", columns="verb", aggfunc="mean")
    pivot = pivot.reindex(columns=verb_order)
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax)
    ax.set_title("Mean Certainty: Verb × Statement Type")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "exp4_factive_verb.png", bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {PLOTS_DIR / 'exp4_factive_verb.png'}")

    return {
        "n_valid": len(df),
        "kruskal_wallis_p": p_kw,
        "certainty_by_verb": {v: df[df["verb"] == v]["certainty_level"].mean() for v in verb_order},
        "truth_by_verb": {v: df[df["verb"] == v]["implies_truth"].mean() for v in verb_order},
    }


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY FIGURE
# ─────────────────────────────────────────────────────────────────────────────
def create_summary_figure(exp1_stats, exp2_stats, exp3_stats, exp4_stats):
    """Create a single summary figure with key findings across all experiments."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Do LLMs Differentiate Epistemic from Non-Epistemic Beliefs?", fontsize=14, fontweight="bold")

    # Exp 1: Classification accuracy
    ax = axes[0, 0]
    types = ["Epistemic", "Non-Epistemic"]
    accs = [exp1_stats["epistemic_accuracy"], exp1_stats["non_epistemic_accuracy"]]
    bars = ax.bar(types, accs, color=["#4C72B0", "#DD8452"], edgecolor="black")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
    ax.set_title(f"Exp 1: Classification Accuracy\n(Kappa={exp1_stats['cohens_kappa']:.2f})")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.1)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{acc:.2f}", ha="center")
    ax.legend()

    # Exp 2: Response features
    ax = axes[0, 1]
    data2 = load_results("exp2_differential_response.json")
    rows2 = []
    for d in data2:
        c = d["coding"]
        if c.get("agreement_level") is not None:
            rows2.append({"belief_type": d["belief_type"], "uses_evidence": bool(c.get("uses_evidence")),
                          "uses_values": bool(c.get("uses_values")), "challenges": bool(c.get("challenges"))})
    df2 = pd.DataFrame(rows2)
    feats = ["uses_evidence", "uses_values", "challenges"]
    x = np.arange(len(feats))
    width = 0.35
    ep_means = [df2[df2["belief_type"] == "epistemic"][f].mean() for f in feats]
    ne_means = [df2[df2["belief_type"] == "non_epistemic"][f].mean() for f in feats]
    ax.bar(x - width/2, ep_means, width, label="Epistemic", color="#4C72B0", edgecolor="black")
    ax.bar(x + width/2, ne_means, width, label="Non-Epistemic", color="#DD8452", edgecolor="black")
    ax.set_title("Exp 2: Spontaneous Response Patterns")
    ax.set_ylabel("Proportion")
    ax.set_xticks(x)
    ax.set_xticklabels(["Uses Evidence", "Uses Values", "Challenges"])
    ax.legend()

    # Exp 3: Revision strength
    ax = axes[1, 0]
    data3 = load_results("exp3_belief_revision.json")
    rows3 = []
    for d in data3:
        c = d["coding"]
        if c.get("revision_strength") is not None:
            rows3.append({"belief_type": d["belief_type"], "revision_strength": int(c["revision_strength"])})
    df3 = pd.DataFrame(rows3)
    sns.boxplot(data=df3, x="belief_type", y="revision_strength", ax=ax,
                palette=["#4C72B0", "#DD8452"], order=["epistemic", "non_epistemic"])
    ax.set_title(f"Exp 3: Revision Strength\n(Cohen's d={exp3_stats['cohens_d']:.2f}, p={exp3_stats['mann_whitney_p']:.4f})")
    ax.set_xlabel("Belief Type")
    ax.set_ylabel("Revision Strength (1=maintain, 5=revise)")

    # Exp 4: Verb sensitivity
    ax = axes[1, 1]
    cert = exp4_stats["certainty_by_verb"]
    verbs = list(cert.keys())
    vals = list(cert.values())
    bars = ax.bar(verbs, vals, color=["#4C72B0", "#55A868", "#DD8452"], edgecolor="black")
    ax.set_title(f"Exp 4: Certainty by Verb\n(Kruskal-Wallis p={exp4_stats['kruskal_wallis_p']:.4f})")
    ax.set_ylabel("Mean Certainty Level")
    ax.set_xlabel("Verb Used")
    ax.set_ylim(0, 5.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f"{v:.2f}", ha="center")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "summary_all_experiments.png", bbox_inches="tight")
    plt.close()
    print(f"\nSummary plot saved: {PLOTS_DIR / 'summary_all_experiments.png'}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    exp1_stats = analyze_exp1()
    exp2_stats = analyze_exp2()
    exp3_stats = analyze_exp3()
    exp4_stats = analyze_exp4()

    create_summary_figure(exp1_stats, exp2_stats, exp3_stats, exp4_stats)

    # Save summary stats
    summary = {
        "experiment_1_classification": exp1_stats,
        "experiment_3_revision": exp3_stats,
        "experiment_4_verb_sensitivity": exp4_stats,
    }
    with open(RESULTS_DIR / "summary_stats.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nAll analysis complete. Summary saved to {RESULTS_DIR / 'summary_stats.json'}")
