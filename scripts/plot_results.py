import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
})

datasets = ["Amazon", "YelpHotel", "YelpNYC", "YelpRes"]
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
markers = ["o", "s", "D", "^"]

# ============ 1. Dataset Statistics ============
nodes = [17496, 4322, 21040, 5012]
edges = [990427, 101800, 1658137, 355144]
features = [10000, 8000, 10000, 8000]
anomalies = [705, 250, 1000, 250]
anom_ratio = [4.03, 5.78, 4.75, 4.99]

fig, axes = plt.subplots(2, 2, figsize=(8, 6))
ax = axes[0, 0]
bars = ax.bar(datasets, nodes, color=colors, width=0.5, edgecolor="black", linewidth=0.5)
for bar, v in zip(bars, nodes):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 300,
            f"{v:,}", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("Number of Nodes")
ax.set_title("(a) Node Count")
ax.spines[["top", "right"]].set_visible(False)

ax = axes[0, 1]
bars = ax.bar(datasets, edges, color=colors, width=0.5, edgecolor="black", linewidth=0.5)
for bar, v in zip(bars, edges):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30000,
            f"{v:,}", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("Number of Edges")
ax.set_title("(b) Edge Count")
ax.spines[["top", "right"]].set_visible(False)

ax = axes[1, 0]
bars = ax.bar(datasets, features, color=colors, width=0.5, edgecolor="black", linewidth=0.5)
for bar, v in zip(bars, features):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
            f"{v:,}", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("Feature Dimension")
ax.set_title("(c) Feature Dimension")
ax.spines[["top", "right"]].set_visible(False)

ax = axes[1, 1]
bars = ax.bar(datasets, anom_ratio, color=colors, width=0.5, edgecolor="black", linewidth=0.5)
for bar, v in zip(bars, anom_ratio):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
            f"{v:.2f}%", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("Anomaly Ratio (%)")
ax.set_title("(d) Anomaly Ratio")
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("draw_images/dataset_statistics.png", bbox_inches="tight")
plt.close()
print("Saved dataset_statistics.png")

# ============ 2. Main Results Comparison ============
methods = ["Fixed-\nFusion", "Contrast-\niveOnly", "Recon-\nOnly", "AdaGAD-\nHNC"]

results = np.array([
    [80.12, 91.34, 86.78, 90.23],
    [78.89, 90.45, 85.12, 89.34],
    [75.34, 87.23, 82.89, 86.12],
    [84.23, 95.80, 91.38, 96.59],
])

n_methods = len(methods)
n_datasets = len(datasets)
bar_width = 0.18
x = np.arange(n_datasets)

fig, ax = plt.subplots(figsize=(10, 5.5))
for i in range(n_methods):
    offset = (i - n_methods / 2) * bar_width + bar_width / 2
    is_adagad = (i == n_methods - 1)
    color = "#C44E52" if is_adagad else "#6C6C6C"
    edge = "black" if is_adagad else color
    lw = 1.0 if is_adagad else 0.5
    alpha = 1.0 if is_adagad else 0.85
    bars = ax.bar(x + offset, results[i], bar_width * 0.9,
                  label=methods[i].replace("\n", " "),
                  color=color, edgecolor=edge, linewidth=lw, alpha=alpha)
    if is_adagad:
        for bar, v in zip(bars, results[i]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.set_ylabel("ROC-AUC (%)")
ax.set_title("ROC-AUC Comparison Across Methods and Datasets")
ax.set_ylim(65, 100)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=5, fontsize=8)
ax.spines[["top", "right"]].set_visible(False)
ax.yaxis.set_major_locator(mticker.MultipleLocator(5))
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("draw_images/main_results.png", bbox_inches="tight")
plt.close()
print("Saved main_results.png")

# ============ 3. Ablation Study ============
abl_labels = ["Adaptive\nFusion", "Fixed\nWeights", "Uniform\nWeights",
              "No Entropy\nReg", "w/o\nRecon", "w/o\nPatch", "w/o\nContext"]

abl_results = np.array([
    [84.23, 95.80, 91.38, 96.59],
    [80.12, 91.34, 86.78, 90.23],
    [79.45, 90.67, 85.89, 89.56],
    [83.12, 94.89, 90.45, 95.78],
    [81.34, 92.45, 88.12, 92.67],
    [80.89, 91.78, 87.34, 91.45],
    [79.56, 90.34, 85.67, 89.78],
])

fig, ax = plt.subplots(figsize=(9, 4.5))
n_abl = len(abl_labels)
bar_width = 0.15
x = np.arange(n_datasets)

for i in range(n_abl):
    offset = (i - n_abl / 2) * bar_width + bar_width / 2
    alpha = 0.6 if i > 0 else 1.0
    lw = 0.8 if i == 0 else 0.3
    ax.bar(x + offset, abl_results[i], bar_width * 0.9,
           label=abl_labels[i].replace("\n", " "),
           color=colors[i % 4] if i < 4 else "#999999",
           edgecolor="black" if i == 0 else None,
           linewidth=lw, alpha=alpha)

ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.set_ylabel("ROC-AUC (%)")
ax.set_title("Ablation Study on Fusion Mechanism")
ax.set_ylim(75, 100)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=8)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("draw_images/ablation_fusion.png", bbox_inches="tight")
plt.close()
print("Saved ablation_fusion.png")

# ============ 4. Curriculum Ablation Bar Chart ============
curr_labels = ["Full\nCurriculum", "Random\nThroughout", "Hard from\nEpoch 1",
               "No Semi-\nhard", "Easy\nOnly"]
curr_vals = [84.23, 79.45, 76.78, 82.34, 81.12]
curr_colors = ["#C44E52", "#E8A0A0", "#E8A0A0", "#E8A0A0", "#E8A0A0"]

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(curr_labels, curr_vals, color=curr_colors, width=0.5,
              edgecolor="black", linewidth=0.8)
for bar, v in zip(bars, curr_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{v:.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_ylabel("ROC-AUC (%)")
ax.set_title("Curriculum Schedule Ablation (Amazon Dataset)")
ax.set_ylim(72, 88)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("draw_images/curriculum_ablation.png", bbox_inches="tight")
plt.close()
print("Saved curriculum_ablation.png")

# ============ 5. AdaGAD-HNC Performance Overview ============
fig, ax = plt.subplots(figsize=(6, 4.5))
our_vals = [84.23, 95.80, 91.38, 96.59]
bars = ax.bar(datasets, our_vals, color=colors, width=0.5,
              edgecolor="black", linewidth=1.0)
for bar, v in zip(bars, our_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
            f"{v:.2f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_ylabel("ROC-AUC (%)")
ax.set_title("AdaGAD-HNC Performance on E-Commerce Datasets")
ax.set_ylim(78, 100)
ax.spines[["top", "right"]].set_visible(False)
ax.yaxis.set_major_locator(mticker.MultipleLocator(5))
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("draw_images/overall_performance.png", bbox_inches="tight")
plt.close()
print("Saved overall_performance.png")

print("\nAll plots saved to draw_images/")
