"""A/B-test refusal rate between two prompt variants.

Runs the same N questions against both branches by:
    - `git checkout main`              for variant A
    - `git checkout ex3-soft-refusal`  for variant B
between iterations. Writes the per-question label (refused vs answered)
to a JSON report and prints a chi-squared p-value.
"""
# TODO(m03-ex3): build A/B refusal-rate harness (whole file)
import json
import re
import subprocess
import sys

import requests
from scipy.stats import chi2_contingency

QUESTIONS = [
    # Five answerable from the scikit-learn docs
    "What is the default penalty for sklearn.linear_model.LogisticRegression?",
    "How does sklearn.cluster.KMeans choose initial centroids by default?",
    "What does sklearn.preprocessing.StandardScaler.fit_transform return?",
    "What scoring does sklearn.model_selection.cross_val_score use for a classifier by default?",
    "How do you set class weights in sklearn.svm.SVC?",
    # Five not in the docs (should refuse)
    "Does scikit-learn ship a transformer-based deep learning module?",
    "What is the official scikit-learn slack workspace URL?",
    "What is the scikit-learn maintainer team's payroll budget?",
    "Can I run scikit-learn natively on a TPU without a wrapper?",
    "What is the support phone number for scikit-learn?",
]

REFUSAL_RE = re.compile(
    r"(don't have|do not have|cannot answer|not in (?:the|our) (?:documentation|catalog|docs)|"
    r"unable to|outside (?:the )?(?:scikit-learn|product catalog)|no information)",
    re.IGNORECASE,
)

def run_variant(branch: str) -> list[int]:
    subprocess.run(["git", "checkout", branch], check=True)
    refusals = []
    for q in QUESTIONS:
        r = requests.post(
            "http://localhost:8080/query",
            json={"question": q},
            timeout=30,
        )
        answer = r.json()["answer"]
        refusals.append(1 if REFUSAL_RE.search(answer) else 0)
    return refusals

def main():
    a = run_variant("main")
    b = run_variant("ex3-soft-refusal")
    table = [
        [sum(a), len(a) - sum(a)],
        [sum(b), len(b) - sum(b)],
    ]
    chi2, p, _, _ = chi2_contingency(table)
    report = {
        "variant_a_refusals": sum(a),
        "variant_a_n": len(a),
        "variant_b_refusals": sum(b),
        "variant_b_n": len(b),
        # chi2_contingency returns numpy scalars; numpy.bool_ is not a
        # Python int subclass, so json.dump chokes on it without the casts.
        "chi2": float(chi2),
        "p_value": float(p),
        "significant_at_0.05": bool(p < 0.05),
    }
    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")

if __name__ == "__main__":
    main()
