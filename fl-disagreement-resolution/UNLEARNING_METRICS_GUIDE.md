# Unlearning Metrics Interpretation Guide

## Overview

This guide explains how to interpret the unlearning metrics. Forget accuracy alone does not indicate unlearning success. The metrics below are meant to be read together.

---

## Metrics Reference Table

| Metric | Good Unlearning | Poor Unlearning | What It Measures |
|--------|----------------|-----------------|-------------------|
| **forget_accuracy** | 70-98% | N/A | Accuracy on forgotten data (can be high via generalization) |
| **activation_cosine_similarity** | <0.6 | >0.9 | Similarity of feature representations (lower = more different) |
| **forget_confidence_mean_unlearned** | 0.6-0.8 | >0.95 | Model's certainty on forgotten data |
| **forget_entropy_mean_unlearned** | >0.7 | <0.3 | Prediction uncertainty (higher = less certain) |
| **js_divergence_mean** | 0.2-0.5 | <0.05 | Difference in prediction distributions |
| **mia_accuracy** | ~0.5 | >0.7 | Membership inference attack success rate |
| **unlearning_score** | >0 | <0 | forget_accuracy_original - forget_accuracy_unlearned |

---

## Why Can Forget Accuracy Be High After Unlearning?

### Example: high forget accuracy after unlearning

When you see results like:
```json
{
  "forget_accuracy_original": 0.928,
  "forget_accuracy_unlearned": 0.974,
  "unlearning_score": -0.046
}
```

This looks like unlearning **failed** (accuracy improved on forgotten data!), but it may actually indicate **successful generalization**:

### Explanation:

1. **Original model (trained on ALL clients 0-9):**
   - Learns features from 10,000 samples
   - Some overfitting possible
   - Accuracy on forget set: 92.8%

2. **Unlearned model (trained on ONLY clients 5-9):**
   - Learns from 5,000 cleaner samples
   - Less overfitting
   - Better generalization to forgotten data
   - Accuracy on forget set: 97.4%

3. **Why this is OK:**
   - Model hasn't memorized the forgotten data
   - It learned general patterns that transfer
   - Evidence: MIA accuracy ≈ 0.5 (random guessing)

### How to Verify Good Unlearning Despite High Accuracy:

Check these metrics together:

- **Low cosine similarity** (<0.6): Model uses different features
- **Lower confidence** (<0.8): Model is less certain on forgotten data
- **Higher entropy** (>0.7): Predictions are more uncertain
- **JS divergence** (>0.2): Predictions differ from original model
- **MIA ≈ 0.5**: Cannot detect membership

---

## Example: Good Unlearning with High Forget Accuracy

```json
{
  "forget_accuracy_original": 0.928,
  "forget_accuracy_unlearned": 0.974,
  "unlearning_score": -0.046,

  "activation_cosine_similarity": 0.42,         // OK Low - different features

  "forget_confidence_mean_original": 0.95,      // Original was very certain
  "forget_confidence_mean_unlearned": 0.72,     // OK Now less certain

  "forget_entropy_mean_original": 0.18,         // Original had low entropy
  "forget_entropy_mean_unlearned": 0.85,        // OK Now high entropy

  "js_divergence_mean": 0.31,                   // OK Predictions differ

  "mia_accuracy_original": 0.495,               // OK Random guessing
  "mia_accuracy_unlearned": 0.493               // OK Still random
}
```

**Interpretation:** Model has successfully unlearned! It's using different features (low cosine similarity), is less confident (lower confidence, higher entropy), makes different predictions (JS divergence), and doesn't leak membership information (MIA ≈ 0.5). The high accuracy is due to generalization, not memorization.

---

## Example: Poor Unlearning

```json
{
  "forget_accuracy_original": 0.928,
  "forget_accuracy_unlearned": 0.925,
  "unlearning_score": 0.003,

  "activation_cosine_similarity": 0.94,         // Bad High - very similar features

  "forget_confidence_mean_unlearned": 0.96,     // Bad Still very certain
  "forget_entropy_mean_unlearned": 0.21,        // Bad Still low entropy

  "js_divergence_mean": 0.03,                   // Bad Almost identical

  "mia_accuracy_unlearned": 0.72                // Bad Can detect membership!
}
```

**Interpretation:** Unlearning likely failed. Model is using similar features (high cosine similarity), is still confident, makes similar predictions, and leaks membership information.

---

## Dataset-Specific Considerations

### MNIST (Image Classification)
- **Characteristics:** Simple patterns, high inter-class separation
- **Expected:** High forget_accuracy even after unlearning
- **Focus on:** Cosine similarity, confidence, entropy

### N-CMAPSS (Time-Series RUL)
- **Characteristics:** Complex temporal patterns, client-specific degradation
- **Expected:** Lower forget_accuracy after unlearning
- **Focus on:** RMSE difference, prediction variance

### Tabular (Synthetic Classification)
- **Characteristics:** Depends on IID vs non-IID distribution
- **Expected:** Moderate forget_accuracy changes
- **Focus on:** All metrics combined

---

## Strategy-Specific Interpretation

### Exact Retraining (gold standard)
- **Expected:** Best unlearning completeness
- **Metrics:** All metrics should show significant change
- **Trade-off:** Slowest, use as baseline

### SISA
- **Expected:** Good completeness, much faster
- **Metrics:** Similar to exact retraining but slightly less complete
- **Trade-off:** Slight accuracy drop for speed gain

### Knowledge Distillation
- **Expected:** Moderate completeness, preserves capacity
- **Metrics:** Lower confidence change than exact retraining
- **Trade-off:** Faster but less complete unlearning

---

## Quick Decision Tree

```
Is forget_accuracy high after unlearning?
├─ YES
│  ├─ Is cosine_similarity < 0.6? ─── YES ─── GOOD (generalization)
│  │                              └─ NO ──── BAD (memorization)
│  │
│  ├─ Is confidence < 0.8? ────────── YES ─── GOOD
│  │                          └────── NO ──── BAD
│  │
│  └─ Is MIA ≈ 0.5? ──────────────── YES ─── GOOD
│                            └────── NO ──── BAD
│
└─ NO
   └─ Is cosine_similarity < 0.6? ─── YES ─── GOOD
                                  └─ NO ──── Note: check other metrics
```

---

## Reporting in Thesis

When presenting results, show:

1. **Primary metrics:** forget_accuracy, unlearning_score
2. **Structural change:** activation_cosine_similarity
3. **Uncertainty:** confidence, entropy
4. **Privacy:** MIA accuracy
5. **Efficiency:** unlearning_time, affected_slices (SISA)

**Example table:**
```
| Strategy | Forget Acc ↓ | Cos Sim ↓ | Confidence ↓ | Entropy ↑ | MIA ≈0.5 | Time (s) |
|----------|-------------|-----------|--------------|----------|----------|----------|
| Exact    | 0.974       | 0.42      | 0.72         | 0.85     | 0.493    | 13.7     |
| SISA     | 0.698       | 0.35      | 0.65         | 0.91     | 0.508    | 13.0     |
| Distill  | 0.950       | 0.48      | 0.78         | 0.72     | 0.494    | 20.6     |
```

---

## Common Misconceptions

- **"Low forget_accuracy means good unlearning"**
-> Not necessarily! The model could just be broken.

- **"High forget_accuracy means bad unlearning"**
-> Not necessarily! Check activation distance and MIA.

- **"SISA should always be fastest"**
-> Only if checkpoints work. Check affected_slices metric.

- **"Good unlearning = structural change + privacy preservation"**
-> Measured by cosine similarity + MIA, not just accuracy.

---

## Troubleshooting

### SISA shows no efficiency gain
**Check:** num_affected_slices vs total_slices in metrics
**Expected:** <50% affected for typical scenarios
**If 100% affected:** Bug in slice selection or checkpoint loading

### All strategies show similar results
**Check:** Are you using MNIST? Try N-CMAPSS or non-IID tabular
**Reason:** Dataset too simple, strong generalization masks differences

### MIA accuracy >> 0.5
**Concern:** Model leaking membership information
**Action:** Increase unlearning strength or epochs

---

## References

- SISA: [Bourtoule et al., 2021, "Machine Unlearning"](https://arxiv.org/abs/1912.03817)
- MIA: [Shokri et al., 2017, "Membership Inference Attacks"](https://arxiv.org/abs/1610.05820)
- FL Unlearning: [Liu et al., 2022, "The Right to be Forgotten in FL"](https://arxiv.org/abs/2203.07320)
