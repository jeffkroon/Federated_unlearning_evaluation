"""
Generate the three thesis result figures directly from the n=10 aggregated data.
Outputs PDF (vector, for LaTeX) + PNG (preview) to ../figures/.
  1. fig_forgetting_utility.pdf  -- forgetting vs utility trade-off (per dataset)
  2. fig_complexity.pdf          -- utility per strategy across the complexity axis
  3. fig_seed_spread.pdf         -- per-seed MNIST utility (stability / uncertainty)
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "figures")
os.makedirs(OUT, exist_ok=True)

agg = pd.read_csv(os.path.join(HERE, "aggregated_mean_std.csv"))
per = pd.read_csv(os.path.join(HERE, "per_seed_values.csv"))

#consistent strategy order, labels, colours
STRATS = ["exact_retraining", "federated_exact_retraining", "sisa", "distillation", "mf"]
LABEL = {
    "exact_retraining": "Centralized exact RT",
    "federated_exact_retraining": "Federated exact RT",
    "sisa": "SISA",
    "distillation": "Distillation",
    "mf": "MF",
}
SHORT = {
    "exact_retraining": "Cent. exact",
    "federated_exact_retraining": "Fed. exact",
    "sisa": "SISA",
    "distillation": "Distill.",
    "mf": "MF",
}
COLOR = {
    "exact_retraining": "#1f77b4",
    "federated_exact_retraining": "#7fc7ff",
    "sisa": "#2ca02c",
    "distillation": "#e6a817",
    "mf": "#d62728",
}
DATASETS = ["mnist", "adult", "cifar10"]
DSLABEL = {"mnist": "MNIST", "adult": "Adult Income", "cifar10": "CIFAR-10"}

plt.rcParams.update({"font.size": 11})


def get(ds, st, col):
    row = agg[(agg.dataset == ds) & (agg.strategy == st)]
    return float(row[col].values[0]) if len(row) else float("nan")


#---------------------------------------------------------------- figure 1
# Forgetting (unlearning score) vs utility, one panel per dataset.
fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True)
handles = None
for ax, ds in zip(axes, DATASETS):
    hs = []
    for st in STRATS:
        x = get(ds, st, "unlearning_score_mean")
        y = get(ds, st, "utility_accuracy_test_mean")
        h = ax.scatter(x, y, s=150, color=COLOR[st], edgecolor="black",
                       linewidth=0.9, zorder=3, label=LABEL[st])
        hs.append(h)
    handles = hs
    ds_is_n1 = "  (n=1)" if ds == "cifar10" else ""
    ax.set_title(DSLABEL[ds] + ds_is_n1)
    ax.set_xlabel("Unlearning score  (more forgetting →)")
    ax.grid(alpha=0.25, zorder=0)
    ax.margins(x=0.18)
axes[0].set_ylabel("Utility (test accuracy)")
axes[0].set_ylim(0, 1.05)
fig.suptitle("Forgetting versus utility per strategy", y=1.02, fontsize=13)
fig.legend(handles=handles, labels=[LABEL[s] for s in STRATS],
           loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(os.path.join(OUT, "fig_forgetting_utility.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig_forgetting_utility.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- figure 2
# Utility per strategy across the three datasets (complexity axis).
fig, ax = plt.subplots(figsize=(7.5, 4.6))
xpos = list(range(len(DATASETS)))
for st in STRATS:
    ys = [get(ds, st, "utility_accuracy_test_mean") for ds in DATASETS]
    es = [get(ds, st, "utility_accuracy_test_std") for ds in DATASETS]
    es = [0 if pd.isna(e) else e for e in es]
    ax.errorbar(xpos, ys, yerr=es, marker="o", markersize=8, capsize=3,
                linewidth=2, color=COLOR[st], label=LABEL[st],
                markeredgecolor="black", markeredgewidth=0.6)
ax.set_xticks(xpos)
ax.set_xticklabels(["MNIST\n(near-saturated)", "Adult Income\n(low-dim tabular)",
                    "CIFAR-10\n(high-dim RGB)"])
ax.set_xlabel("Increasing task complexity →")
ax.set_ylabel("Utility (test accuracy)")
ax.set_ylim(0, 1.05)
ax.set_title("Utility across datasets of increasing complexity")
ax.grid(alpha=0.25, axis="y")
ax.legend(fontsize=9, loc="lower left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_complexity.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig_complexity.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

#---------------------------------------------------------------- figure 3
#Per-seed MNIST utility: stability of conservative methods vs MF spread.
mn = per[(per.dataset == "mnist") & (per.metric == "utility_accuracy_test")]
fig, ax = plt.subplots(figsize=(8, 4.6))
order = ["federated_exact_retraining", "distillation", "sisa", "exact_retraining", "mf"]
for i, st in enumerate(order):
    vals = mn[mn.strategy == st]["value"].astype(float).values
    ax.scatter([i] * len(vals), vals, s=90, color=COLOR[st],
               edgecolor="black", linewidth=0.7, alpha=0.85, zorder=3)
    if len(vals):
        m = vals.mean()
        ax.hlines(m, i - 0.22, i + 0.22, color="black", linewidth=2, zorder=4)
ax.set_xticks(range(len(order)))
ax.set_xticklabels([LABEL[s].replace(" ", "\n", 1) for s in order])
ax.set_ylabel("Utility (test accuracy)")
ax.set_ylim(0, 1.05)
ax.set_title("Per-seed MNIST utility across ten seeds (42-51)\n"
             "each dot = one seed; bar = mean")
ax.grid(alpha=0.25, axis="y")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_seed_spread.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "fig_seed_spread.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- sanity print
print("VERIFY against tables:")
for ds in DATASETS:
    for st in STRATS:
        print(f"  {ds:7s} {st:26s} util={get(ds,st,'utility_accuracy_test_mean'):.3f} "
              f"unl={get(ds,st,'unlearning_score_mean'):.3f}")
print("MNIST MF per-seed values:",
      sorted(mn[mn.strategy=='mf']['value'].astype(float).round(3).tolist()))
print("DONE -> figures/fig_forgetting_utility, fig_complexity, fig_seed_spread")
