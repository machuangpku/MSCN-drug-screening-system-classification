import os
import random
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as ticker

from matplotlib.lines import Line2D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# ==========================================
# 0. Reproducibility
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ==========================================
# 1. Device
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 2. Font and Display Settings
# ==========================================
font_path = 'times.ttf'
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    custom_font = fm.FontProperties(fname=font_path)
    matplotlib.rcParams['font.family'] = custom_font.get_name()
    print(f"Successfully loaded local font: {font_path}")
else:
    print(f"Warning: '{font_path}' was not found. Using default font.")

matplotlib.rcParams['axes.unicode_minus'] = False

# ==========================================
# 3. Custom Class Names
# ==========================================
CLASS_NAMES = {
    0: "G1 phase arrest",
    1: "Homeostasis disruption",
    2: "Apoptosis"
}

SUBCLASS_NAMES = {
    1: {0: "Subtype 1", 1: "Subtype 2"},
    2: {0: "Subtype 1", 1: "Subtype 2"}
}

# ==========================================
# 4. Data Loading
# ==========================================
excel_file = "five_zjh 0424.xlsx"

try:
    df = pd.read_excel(excel_file)
except FileNotFoundError:
    print(f"Error: Could not find '{excel_file}'.")
    raise SystemExit


class_0_data = df.iloc[:, 0].dropna().values

class_1_sub0 = df.iloc[:, 4].dropna().values
class_1_sub1 = df.iloc[:, 5].dropna().values

class_2_sub0 = df.iloc[:, 8].dropna().values
class_2_sub1 = df.iloc[:, 9].dropna().values

X = np.concatenate([
    class_0_data,
    class_1_sub0, class_1_sub1,
    class_2_sub0, class_2_sub1
]).reshape(-1, 1)


y_major = np.concatenate([
    np.full(len(class_0_data), 0),
    np.full(len(class_1_sub0), 1),
    np.full(len(class_1_sub1), 1),
    np.full(len(class_2_sub0), 2),
    np.full(len(class_2_sub1), 2)
]).astype(int)


y_sub = np.concatenate([
    np.full(len(class_0_data), -1),
    np.full(len(class_1_sub0), 0),
    np.full(len(class_1_sub1), 1),
    np.full(len(class_2_sub0), 0),
    np.full(len(class_2_sub1), 1)
]).astype(int)

print(f"Total samples: {len(X)}")
print(f"Class counts: "
      f"class0={np.sum(y_major == 0)}, "
      f"class1={np.sum(y_major == 1)}, "
      f"class2={np.sum(y_major == 2)}")

# ==========================================
# 5. Train/Test Split + Scaling
# ==========================================
X_train, X_test, y_train, y_test, y_sub_train, y_sub_test = train_test_split(
    X, y_major, y_sub,
    test_size=0.4,
    random_state=42,
    stratify=y_major
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# -------------------------------

# -------------------------------
X_test = scaler.transform(X_test)
X_all = scaler.transform(X)

X_train_tensor = torch.FloatTensor(X_train).to(device)
X_test_tensor = torch.FloatTensor(X_test).to(device)
X_all_tensor = torch.FloatTensor(X_all).to(device)

y_train_tensor = torch.LongTensor(y_train).to(device)
y_test_tensor = torch.LongTensor(y_test).to(device)

y_sub_train_tensor = torch.LongTensor(y_sub_train).to(device)

# ==========================================
# 6. Model
# ==========================================
class CompactTSNENet(nn.Module):
    def __init__(self):
        super(CompactTSNENet, self).__init__()
        self.layer1 = nn.Linear(1, 16)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(16, 8)
        self.relu2 = nn.ReLU()
        self.embed = nn.Linear(8, 4)
        self.relu3 = nn.ReLU()
        self.classifier = nn.Linear(4, 3)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        feat = self.embed(x)
        feat = self.relu3(feat)
        logits = self.classifier(feat)
        return logits, feat

# ==========================================
# 7. Center Loss
# ==========================================
class CenterLoss(nn.Module):
    def __init__(self, num_classes=3, feat_dim=4):
        super(CenterLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, features, labels):
        batch_centers = self.centers[labels]
        loss = ((features - batch_centers) ** 2).sum(dim=1).mean()
        return loss

# ==========================================
# 8. Subtype Merge Loss

# ==========================================
def subtype_merge_loss(features, y_major_tensor, y_sub_tensor):
    total_loss = torch.tensor(0.0, device=features.device)
    count = 0

    for major_class in [1, 2]:
        mask0 = (y_major_tensor == major_class) & (y_sub_tensor == 0)
        mask1 = (y_major_tensor == major_class) & (y_sub_tensor == 1)

        if mask0.sum() > 0 and mask1.sum() > 0:
            center0 = features[mask0].mean(dim=0)
            center1 = features[mask1].mean(dim=0)
            total_loss = total_loss + ((center0 - center1) ** 2).sum()
            count += 1

    if count == 0:
        return torch.tensor(0.0, device=features.device)

    return total_loss / count

# ==========================================
# 9. Plot Settings
# ==========================================
colors_plot = {
    0: '#B47575',
    1: '#789B84',
    2: '#7A96BA'
}

marker_map = {
    -1: 'o',
    0: 's',
    1: '^'
}

AXES_LINE_WIDTH = 2.0

plot_groups = [
    {"major": 0, "sub": -1, "label": CLASS_NAMES[0]},
    {"major": 1, "sub": 0, "label": f"{CLASS_NAMES[1]} - {SUBCLASS_NAMES[1][0]}"},
    {"major": 1, "sub": 1, "label": f"{CLASS_NAMES[1]} - {SUBCLASS_NAMES[1][1]}"},
    {"major": 2, "sub": 0, "label": f"{CLASS_NAMES[2]} - {SUBCLASS_NAMES[2][0]}"},
    {"major": 2, "sub": 1, "label": f"{CLASS_NAMES[2]} - {SUBCLASS_NAMES[2][1]}"}
]

legend_handles = [
    Line2D([0], [0], marker='o', color='w', label='G1 phase arrest',
           markerfacecolor=colors_plot[0], markeredgecolor=colors_plot[0], markersize=11),
    Line2D([0], [0], marker='s', color='w', label='Homeostasis disruption - Subtype 1',
           markerfacecolor=colors_plot[1], markeredgecolor=colors_plot[1], markersize=11),
    Line2D([0], [0], marker='^', color='w', label='Homeostasis disruption - Subtype 2',
           markerfacecolor=colors_plot[1], markeredgecolor=colors_plot[1], markersize=11),
    Line2D([0], [0], marker='s', color='w', label='Apoptosis - Subtype 1',
           markerfacecolor=colors_plot[2], markeredgecolor=colors_plot[2], markersize=11),
    Line2D([0], [0], marker='^', color='w', label='Apoptosis - Subtype 2',
           markerfacecolor=colors_plot[2], markeredgecolor=colors_plot[2], markersize=11),
]

# ==========================================
# 10. Output Directory
# ==========================================
output_dir = "tsne_scan_outputs"
os.makedirs(output_dir, exist_ok=True)

# ==========================================
# 11. Training + T-SNE function
# ==========================================
def run_experiment(lambda_center, lambda_merge, perplexity, epochs=400):
    set_seed(42)

    model = CompactTSNENet().to(device)
    center_loss_fn = CenterLoss(num_classes=3, feat_dim=4).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_model = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    optimizer_center = optim.Adam(center_loss_fn.parameters(), lr=0.01)

    # -------------------------------
    # Train
    # -------------------------------
    for epoch in range(epochs):
        model.train()

        logits, features = model(X_train_tensor)

        ce_loss = criterion(logits, y_train_tensor)
        c_loss = center_loss_fn(features, y_train_tensor)
        m_loss = subtype_merge_loss(features, y_train_tensor, y_sub_train_tensor)

        loss = ce_loss + lambda_center * c_loss + lambda_merge * m_loss

        optimizer_model.zero_grad()
        optimizer_center.zero_grad()
        loss.backward()
        optimizer_model.step()
        optimizer_center.step()

    # -------------------------------
    # Evaluate
    # -------------------------------
    model.eval()
    with torch.no_grad():

        test_outputs, _ = model(X_test_tensor)
        test_predicted = torch.argmax(test_outputs, dim=1)
        test_correct = (test_predicted == y_test_tensor).sum().item()
        test_accuracy = test_correct / len(y_test_tensor)


        all_outputs, hidden_features = model(X_all_tensor)
        all_predicted = torch.argmax(all_outputs, dim=1)
        y_all_tensor = torch.LongTensor(y_major).to(device)
        all_correct = (all_predicted == y_all_tensor).sum().item()
        all_accuracy = all_correct / len(y_all_tensor)

    # -------------------------------
    # Extract all features for T-SNE

    # -------------------------------

    hidden_numpy = hidden_features.detach().cpu().numpy().astype(np.float64)


    hidden_numpy = StandardScaler().fit_transform(hidden_numpy)

    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=perplexity,
        init='pca',
        learning_rate='auto',
        early_exaggeration=10
    )

    X_embedded = tsne.fit_transform(hidden_numpy)

    return {
        "lambda_center": lambda_center,
        "lambda_merge": lambda_merge,
        "perplexity": perplexity,
        "test_accuracy": test_accuracy,
        "all_accuracy": all_accuracy,
        "X_embedded": X_embedded
    }

# ==========================================
# 12. Single Plot Saver
# ==========================================
def save_single_plot(result):
    lc = result["lambda_center"]
    lm = result["lambda_merge"]
    p = result["perplexity"]
    test_acc = result["test_accuracy"]
    all_acc = result["all_accuracy"]
    X_embedded = result["X_embedded"]

    fig, ax = plt.subplots(figsize=(7, 7))

    for group in plot_groups:
        major = group["major"]
        sub = group["sub"]
        label = group["label"]

        mask = (y_major == major) & (y_sub == sub)

        if np.sum(mask) == 0:
            continue

        marker = marker_map[sub]
        size = 60 if marker != '*' else 95

        ax.scatter(
            X_embedded[mask, 0],
            X_embedded[mask, 1],
            c=colors_plot[major],
            label=label,
            s=size,
            alpha=0.85,
            marker=marker
        )

    ax.set_title(
        f'T-SNE Features\nlc={lc}, lm={lm}, p={p}, '
        f'all_acc={all_acc:.3f}, test_acc={test_acc:.3f}',
        fontsize=18
    )
    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_linewidth(AXES_LINE_WIDTH)

    ax.tick_params(
        axis='both',
        which='major',
        labelsize=12,
        width=AXES_LINE_WIDTH,
        length=AXES_LINE_WIDTH * 3
    )

    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))

    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=12)

    png_filename = os.path.join(output_dir, f'tsne_lc{lc}_lm{lm}_p{p}.png')
    pdf_filename = os.path.join(output_dir, f'tsne_lc{lc}_lm{lm}_p{p}.pdf')

    plt.savefig(png_filename, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_filename, format='pdf', bbox_inches='tight')
    plt.close(fig)

    return png_filename, pdf_filename

# ==========================================
# 13. Parameter Scan
# ==========================================
lambda_center_list = [0.6]
lambda_merge_list = [0.15]
perplexity_list = [15]

results = []

total_runs = len(lambda_center_list) * len(lambda_merge_list) * len(perplexity_list)
run_idx = 0

for lc, lm, p in itertools.product(lambda_center_list, lambda_merge_list, perplexity_list):
    run_idx += 1
    print(f"\nRunning [{run_idx}/{total_runs}] -> "
          f"lambda_center={lc}, lambda_merge={lm}, perplexity={p}")

    result = run_experiment(
        lambda_center=lc,
        lambda_merge=lm,
        perplexity=p,
        epochs=400
    )

    png_path, pdf_path = save_single_plot(result)
    result["png_path"] = png_path
    result["pdf_path"] = pdf_path

    results.append(result)

    print(f"Finished: all_acc={result['all_accuracy']:.4f}, "
          f"test_acc={result['test_accuracy']:.4f}")
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")

# ==========================================
# 14. Save Summary CSV
# ==========================================
summary_records = []
for r in results:
    summary_records.append({
        "lambda_center": r["lambda_center"],
        "lambda_merge": r["lambda_merge"],
        "perplexity": r["perplexity"],
        "all_accuracy": r["all_accuracy"],
        "test_accuracy": r["test_accuracy"],
        "png_path": r["png_path"],
        "pdf_path": r["pdf_path"]
    })

summary_df = pd.DataFrame(summary_records)
summary_df = summary_df.sort_values(
    by=["all_accuracy", "test_accuracy", "lambda_center", "lambda_merge", "perplexity"],
    ascending=[False, False, True, True, True]
).reset_index(drop=True)

summary_csv_path = os.path.join(output_dir, "results_summary.csv")
summary_df.to_csv(summary_csv_path, index=False)

print(f"\nSaved summary CSV: {summary_csv_path}")
print("\nTop results:")
print(summary_df.head(10).to_string(index=False))

# ==========================================
# 15. Generate Combined Comparison Figure


# ==========================================
col_combos = list(itertools.product(lambda_merge_list, perplexity_list))
row_values = lambda_center_list

fig, axes = plt.subplots(
    nrows=len(row_values),
    ncols=len(col_combos),
    figsize=(28, 14)
)

if len(row_values) == 1 and len(col_combos) == 1:
    axes = np.array([[axes]])
elif len(row_values) == 1:
    axes = axes.reshape(1, -1)
elif len(col_combos) == 1:
    axes = axes.reshape(-1, 1)


result_dict = {}
for r in results:
    key = (r["lambda_center"], r["lambda_merge"], r["perplexity"])
    result_dict[key] = r

for i, lc in enumerate(row_values):
    for j, (lm, p) in enumerate(col_combos):
        ax = axes[i, j]
        r = result_dict[(lc, lm, p)]
        X_embedded = r["X_embedded"]
        all_acc = r["all_accuracy"]
        test_acc = r["test_accuracy"]

        for group in plot_groups:
            major = group["major"]
            sub = group["sub"]

            mask = (y_major == major) & (y_sub == sub)
            if np.sum(mask) == 0:
                continue

            marker = marker_map[sub]
            size = 26 if marker != '*' else 40

            ax.scatter(
                X_embedded[mask, 0],
                X_embedded[mask, 1],
                c=colors_plot[major],
                s=size,
                alpha=0.82,
                marker=marker
            )

        ax.set_title(
            f'lc={lc}, lm={lm}, p={p}\n'
            f'all={all_acc:.3f}, test={test_acc:.3f}',
            fontsize=12
        )
        ax.grid(False)

        for spine in ax.spines.values():
            spine.set_linewidth(1.3)

        ax.tick_params(
            axis='both',
            which='major',
            labelsize=9,
            width=1.3,
            length=4
        )

        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))

combined_png = os.path.join(output_dir, "tsne_comparison_grid.png")
combined_pdf = os.path.join(output_dir, "tsne_comparison_grid.pdf")

fig.suptitle("T-SNE Parameter Scan Comparison", fontsize=22, y=0.98)
fig.legend(
    handles=legend_handles,
    loc='center left',
    bbox_to_anchor=(0.88, 0.5),
    fontsize=13,
    frameon=True
)

plt.tight_layout(rect=[0.02, 0.02, 0.86, 0.95])
plt.savefig(combined_png, dpi=300, bbox_inches='tight')
plt.savefig(combined_pdf, format='pdf', bbox_inches='tight')
plt.close(fig)

print(f"\nSaved combined comparison figure:")
print(combined_png)
print(combined_pdf)

# ==========================================
# 16. Best Result
# ==========================================
best_result = max(
    results,
    key=lambda x: (x["all_accuracy"], x["test_accuracy"])
)

print("\nBest result (selected by all-data accuracy):")
print(f"  lambda_center = {best_result['lambda_center']}")
print(f"  lambda_merge  = {best_result['lambda_merge']}")
print(f"  perplexity    = {best_result['perplexity']}")
print(f"  all_accuracy  = {best_result['all_accuracy']:.4f}")
print(f"  test_accuracy = {best_result['test_accuracy']:.4f}")
print(f"  png_path      = {best_result['png_path']}")
print(f"  pdf_path      = {best_result['pdf_path']}")

print("\nAll visualisations complete!")
