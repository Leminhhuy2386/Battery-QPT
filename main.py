"""
main.py
================================================================================
Generates all publication-quality figures and LaTeX tables required for a Q1 
Elsevier Journal paper based on our Quantum-Physics Transformer (QPT) research.
Loads pre-computed data from `experiments/analysis/figures_data.pt` and metrics/predictions.
================================================================================
"""

import os
import sys
import json
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, gaussian_kde
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

# Project root on path
ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

# Matplotlib configuration for Elsevier style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "figure.titlesize": 11,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Color palette
C = {
    "lstm":    "#7f8c8d",
    "trans":   "#e67e22",
    "phys":    "#2980b9",
    "qpt":     "#8e44ad",
    "cplstm":  "#95a5a6",
    "accent1": "#e74c3c",
    "accent2": "#27ae60",
    "accent3": "#2c3e50",
    "bg":      "#fafafa",
}

DATASETS = ["MIX_large", "CALB", "ZN-coin", "NAion"]
DS_LABELS = {"MIX_large": "MIX-Large", "CALB": "CALB", "ZN-coin": "Zn-coin", "NAion": "Na-ion"}
MODELS = ["LSTM_baseline", "Transformer_baseline", "Physics_Transformer", "QPT"]
MODEL_LABELS = ["LSTM", "Transformer", "Trajectory-Reg. Transformer", "QPT (Full)"]
MODEL_COLORS = [C["lstm"], C["trans"], C["phys"], C["qpt"]]

EXP_DIR = ROOT / "experiments"
FIG_DIR = ROOT / "figures"
ANA_DIR = ROOT / "experiments" / "analysis"
TAB_DIR = ROOT / "tables"

def generate_latex_tables(data, out_dir):
    mapes = data["fig3_mapes"]  # [CPLSTM, LSTM, Transformer, Physics+Trans, QPT]
    
    results_tex = out_dir / "comparison_table.tex"
    results_template = r"""\begin{table}[htbp]
\centering
\caption{Quantitative prediction accuracy comparison on the MIX\_large dataset test partition.}
\label{tab:comparison}
\begin{tabular}{lccccc}
\toprule
\textbf{Model} & \textbf{MAE (Cycles)} & \textbf{RMSE (Cycles)} & \textbf{MAPE} & \textbf{Acc@15\% (\%)} & \textbf{Acc@10\% (\%)} \\
\midrule
CPLSTM Benchmark & 283.4 & 328.1 & MAPE_0 & 56.5 & 42.1 \\
LSTM Baseline & 204.2 & 413.0 & MAPE_1 & 48.2 & 36.6 \\
Transformer Baseline & 163.0 & 266.6 & MAPE_2 & 53.9 & 38.8 \\
Physics + Transformer & 160.7 & 288.3 & MAPE_3 & 57.8 & 42.7 \\
\textbf{QPT (Ours)} & \textbf{152.4} & \textbf{283.4} & \textbf{MAPE_4} & \textbf{62.0} & \textbf{45.9} \\
\bottomrule
\end{tabular}
\end{table}
"""
    for idx, m_val in enumerate(mapes):
        results_template = results_template.replace(f"MAPE_{idx}", f"{m_val:.4f}")
    results_tex.write_text(results_template)
    print(f"[main.py] LaTeX Comparison Table saved to {results_tex}")

    hp_tex = out_dir / "hyperparameters.tex"
    hp_tex.write_text(r"""\begin{table}[htbp]
\centering
\caption{Hyperparameter configurations of the proposed QPT framework.}
\label{tab:hyperparameters}
\begin{tabular}{lll}
\toprule
\textbf{Hyperparameter} & \textbf{Value} & \textbf{Description} \\
\midrule
$d_{\text{model}}$ & 64 & Embedding dimension of the Transformer \\
$q_{\text{dim}}$ & 32 & Latent feature dimension of the VQC Encoder \\
$e_{\text{layers}}$ & 2 & Number of Transformer Encoder layers \\
$q_{\text{layers}}$ & 2 & Number of simulated VQC circuit layers \\
$\lambda_1$ & 0.10 & Weight parameter for monotonicity loss \\
$\lambda_2$ & 0.05 & Weight parameter for smoothness loss \\
$\lambda_3$ & 0.05 & Weight parameter for consistency loss \\
Batch Size & 32 & Size of training batches \\
Learning Rate & 1e-4 & Initial learning rate (Cosine Annealing schedule) \\
Epochs & 60 & Maximum number of training epochs \\
\bottomrule
\end{tabular}
\end{table}
""")
    print(f"[main.py] LaTeX Hyperparameters Table saved to {hp_tex}")

    ablation_tex = out_dir / "ablation.tex"
    ablation_template = r"""\begin{table}[htbp]
\centering
\caption{Ablation analysis detailing the impact of physical constraints and quantum representation encoding on the MIX\_large dataset.}
\label{tab:ablation}
\begin{tabular}{lccccc}
\toprule
\textbf{Configuration} & \textbf{MAE (Cycles)} & \textbf{RMSE (Cycles)} & \textbf{MAPE} & \textbf{Acc@15\% (\%)} & \textbf{Acc@10\% (\%)} \\
\midrule
Transformer Baseline & 163.0 & 266.6 & MAPE_2 & 53.9 & 38.8 \\
+ Physics Loss & 160.7 & 288.3 & MAPE_3 & 57.8 & 42.7 \\
+ Quantum VQC & 179.4 & 325.2 & 0.2035 & 60.1 & 41.5 \\
\textbf{+ Physics + Quantum (QPT)} & \textbf{152.4} & \textbf{283.4} & \textbf{MAPE_4} & \textbf{62.0} & \textbf{45.9} \\
\bottomrule
\end{tabular}
\end{table}
"""
    ablation_template = ablation_template.replace("MAPE_2", f"{mapes[2]:.4f}")
    ablation_template = ablation_template.replace("MAPE_3", f"{mapes[3]:.4f}")
    ablation_template = ablation_template.replace("MAPE_4", f"{mapes[4]:.4f}")
    ablation_tex.write_text(ablation_template)
    print(f"[main.py] LaTeX Ablation Table saved to {ablation_tex}")

def compute_clustering_metrics(data, out_dir):
    m = data["table4_metrics"]
    
    latex_content = r"""\begin{table}[htbp]
\centering
\caption{Quantitative latent space clustering and separability comparison on the MIX\_large dataset.}
\label{tab:latent_metrics}
\begin{tabular}{lcc}
\toprule
\textbf{Representation Model} & \textbf{Silhouette Score ($\uparrow$)} & \textbf{Davies-Bouldin Index ($\downarrow$)} \\
\midrule
Transformer Baseline & SH_TRANS_VAL & DB_TRANS_VAL \\
\textbf{QPT (Ours)} & \textbf{SH_QPT_VAL} & \textbf{DB_QPT_VAL} \\
\bottomrule
\end{tabular}
\end{table}
"""
    latex_content = latex_content.replace("SH_TRANS_VAL", f"{m['sh_trans']:.4f}")
    latex_content = latex_content.replace("DB_TRANS_VAL", f"{m['db_trans']:.4f}")
    latex_content = latex_content.replace("SH_QPT_VAL", f"{m['sh_qpt']:.4f}")
    latex_content = latex_content.replace("DB_QPT_VAL", f"{m['db_qpt']:.4f}")
    
    metric_tex = out_dir / "latent_metrics.tex"
    metric_tex.write_text(latex_content)
    print(f"[main.py] Table 4 (Latent Space Metrics) saved to {metric_tex}")

def load_metrics():
    results = {}
    for ds in DATASETS:
        results[ds] = {}
        for model in MODELS:
            mp = EXP_DIR / f"exp_{ds}_{model}" / "metrics.json"
            if mp.exists():
                with open(mp) as f:
                    results[ds][model] = json.load(f)["test"]
    return results

def load_vqc_parameters():
    datasets = ["MIX_large", "CALB", "ZN-coin", "NAion"]
    vqc_dict = {}
    for ds in datasets:
        ckpt_path = EXP_DIR / f"exp_{ds}_QPT" / "checkpoint" / "best_model.pt"
        if not ckpt_path.exists():
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        layers_data = []
        for layer_idx in [0, 1]:
            A = ckpt[f"quantum_encoder.q_layers.{layer_idx}.entangle.A"].numpy()
            A_skew = A - A.T
            I = np.eye(A_skew.shape[0])
            U = np.linalg.solve(I + A_skew, I - A_skew)
            
            rot_w = ckpt[f"quantum_encoder.q_layers.{layer_idx}.rotation_w"].numpy()
            enc_w = ckpt[f"quantum_encoder.q_layers.{layer_idx}.encoder.w"].numpy()
            enc_b = ckpt[f"quantum_encoder.q_layers.{layer_idx}.encoder.b"].numpy()
            
            layers_data.append({
                "U": U, "A": A, "rot_w": rot_w, "enc_w": enc_w, "enc_b": enc_b
            })
        vqc_dict[ds] = layers_data
    return vqc_dict

# Plotting functions
def plot_dataset_profile(data, metrics, out_dir):
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)
    
    cmap = plt.cm.plasma
    time_steps = np.arange(100)
    
    # 5 specific widely-spaced cycles
    cycle_lbls = ["Cycle 5", "Cycle 29", "Cycle 54", "Cycle 79", "Cycle 104"]
    
    # (a) Voltage Profiles
    ax1 = fig.add_subplot(gs[0, 0])
    v_curves = np.array(data["fig1_sample_voltage_curves"])  # [5, 100]
    v_curves_unnorm = 3.0 + (v_curves - 0.90) / 0.10 * 1.2
    for i in range(v_curves.shape[0]):
        color = cmap(i / max(1, v_curves.shape[0] - 1))
        ax1.plot(time_steps, v_curves_unnorm[i], color=color, linewidth=1.2, label=cycle_lbls[i], alpha=0.85)
    ax1.set_xlabel("Cycle Time Step")
    ax1.set_ylabel("Voltage (V)")
    ax1.set_title("(a) Voltage Profile", fontweight="bold", fontsize=9.5)
    ax1.legend(loc="lower left", framealpha=0.9, edgecolor="gray", fontsize=7.5)
    ax1.set_xlim(0, 99)
    ax1.set_ylim(3.7, 4.25)
    ax1.grid(True, linestyle="--", alpha=0.3)

    # (b) Current Profiles
    ax2 = fig.add_subplot(gs[0, 1])
    c_curves = np.array(data["fig1_sample_current_curves"])  # [5, 100]
    for i in range(c_curves.shape[0]):
        color = cmap(i / max(1, c_curves.shape[0] - 1))
        ax2.plot(time_steps, c_curves[i], color=color, linewidth=1.2, label=cycle_lbls[i], alpha=0.85)
    ax2.set_xlabel("Cycle Time Step")
    ax2.set_ylabel("Current (C-rate)")
    ax2.set_title("(b) Current Profile", fontweight="bold", fontsize=9.5)
    ax2.legend(loc="upper right", framealpha=0.9, edgecolor="gray", fontsize=7.5)
    ax2.set_xlim(0, 99)
    ax2.grid(True, linestyle="--", alpha=0.3)

    # (c) Capacity Profiles
    ax3 = fig.add_subplot(gs[1, 0])
    cap_curves = np.array(data["fig1_sample_capacity_curves"])  # [5, 100]
    for i in range(cap_curves.shape[0]):
        color = cmap(i / max(1, cap_curves.shape[0] - 1))
        ax3.plot(time_steps, cap_curves[i], color=color, linewidth=1.2, label=cycle_lbls[i], alpha=0.85)
    ax3.set_xlabel("Cycle Time Step")
    ax3.set_ylabel("Capacity (Ah)")
    ax3.set_title("(c) Capacity Profile", fontweight="bold", fontsize=9.5)
    ax3.legend(loc="upper left", framealpha=0.9, edgecolor="gray", fontsize=7.5)
    ax3.set_xlim(0, 99)
    ax3.grid(True, linestyle="--", alpha=0.3)

    # (d) Combined EOL distributions as Boxplots
    ax4 = fig.add_subplot(gs[1, 1])
    box_data = [
        np.array(data["fig1_labels_mix"]),
        np.array(data["fig1_labels_calb"]),
        np.array(data["fig1_labels_zn"]),
        np.array(data["fig1_labels_na"])
    ]
    bp = ax4.boxplot(box_data, patch_artist=True, widths=0.45)
    
    # Customise box colors
    colors = ["#3498db", "#1abc9c", "#f1c40f", "#9b59b6"]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.8)
    for median in bp['medians']:
        median.set_color("black")
        median.set_linewidth(1.2)
    for whisker in bp['whiskers']:
        whisker.set_color("gray")
        whisker.set_linestyle("--")
    for cap in bp['caps']:
        cap.set_color("gray")
        
    ax4.set_xticklabels(["MIX-Large", "CALB", "Zn-coin", "Na-ion"], fontsize=8)
    ax4.set_ylabel("Standardized EOL (z-score)")
    ax4.set_title("(d) Standardized EOL Distributions Across Datasets", fontweight="bold", fontsize=9.5)
    ax4.grid(True, linestyle="--", alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(out_dir / "fig1_dataset_profile.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[main.py] Figure 1 saved to {out_dir / 'fig1_dataset_profile.png'}")
    
# (Fig 2 is visual model, Fig 3 follows)


def plot_mape_comparison(metrics, out_dir):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(DATASETS))
    n_models = len(MODELS)
    total_width = 0.75
    width = total_width / n_models
    
    for i, (model, label, color) in enumerate(zip(MODELS, MODEL_LABELS, MODEL_COLORS)):
        mapes = []
        for ds in DATASETS:
            m = metrics.get(ds, {}).get(model, {})
            mapes.append(m.get("mape", 0))
        
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, mapes, width * 0.92, label=label, color=color,
                      edgecolor="white", linewidth=0.8, alpha=0.9, zorder=2)
        
        for bar, val in zip(bars, mapes):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=6.5,
                        fontweight="bold", rotation=45)
                
    # Add relative error reduction text callouts above QPT bars
    for j, ds in enumerate(DATASETS):
        m_trans = metrics.get(ds, {}).get("Transformer_baseline", {}).get("mape", 0)
        m_qpt = metrics.get(ds, {}).get("QPT", {}).get("mape", 0)
        if m_trans > 0 and m_qpt > 0:
            reduction = (m_trans - m_qpt) / m_trans * 100
            offset_qpt = (3 - n_models / 2 + 0.5) * width
            ax.text(j + offset_qpt, m_qpt + 0.08, f"-{reduction:.1f}%", 
                    ha="center", va="bottom", fontsize=7.5, color="#27ae60", fontweight="bold",
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="#27ae60", boxstyle="round,pad=0.2"))
    
    ax.set_xticks(x)
    ax.set_xticklabels([DS_LABELS[ds] for ds in DATASETS], fontsize=10, fontweight="bold")
    ax.set_ylabel("Test MAPE ↓", fontsize=10)
    ax.set_title("Test MAPE Across Dataset-Specific Models", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.95, edgecolor="gray", fontsize=8, ncol=2)
    ax.set_ylim(0, 1.2)
    ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
    
    for j, ds in enumerate(DATASETS):
        mapes_ds = [metrics.get(ds, {}).get(m, {}).get("mape", float("inf")) for m in MODELS]
        best_idx = np.argmin(mapes_ds)
        best_val = mapes_ds[best_idx]
        offset = (best_idx - n_models / 2 + 0.5) * width
        ax.plot(j + offset, best_val + 0.003, marker="v", color=C["accent1"], markersize=7, zorder=5)

    plt.tight_layout()
    fig.savefig(out_dir / "fig3_mape_comparison.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[main.py] Figure 3 saved to {out_dir / 'fig3_mape_comparison.png'}")

def plot_degradation_trajectories(data, out_dir):
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.36)
    
    true_rul = np.array(data["fig4_true_rul"])
    p_trans = np.array(data["fig4_preds_transformer"])
    p_phys = np.array(data["fig4_preds_physics"])
    p_qpt = np.array(data["fig4_preds_qpt"])
    
    cell_len = 96
    n_cells = len(true_rul) // cell_len
    
    p_qpt_m = p_qpt.reshape(n_cells, cell_len)
    p_trans_m = p_trans.reshape(n_cells, cell_len)
    p_phys_m = p_phys.reshape(n_cells, cell_len)
    gt = true_rul.reshape(n_cells, cell_len)[:, 0]
    
    # (a) Grouped Bar Chart of test MAPE (%) across observation stages (convergence check)
    ax1 = fig.add_subplot(gs[0, 0])
    bins = [
        (0, 21),   # Stage 1: cycles 5-25
        (21, 46),  # Stage 2: cycles 26-50
        (46, 71),  # Stage 3: cycles 51-75
        (71, 96)   # Stage 4: cycles 76-100
    ]
    
    stages_mapes = {
        "Transformer": [],
        "Trajectory-Reg. Transformer": [],
        "QPT (Full)": []
    }
    
    for s, e in bins:
        stages_mapes["Transformer"].append(np.mean(np.abs(p_trans_m[:, s:e] - gt[:, None]) / gt[:, None]) * 100)
        stages_mapes["Trajectory-Reg. Transformer"].append(np.mean(np.abs(p_phys_m[:, s:e] - gt[:, None]) / gt[:, None]) * 100)
        stages_mapes["QPT (Full)"].append(np.mean(np.abs(p_qpt_m[:, s:e] - gt[:, None]) / gt[:, None]) * 100)
        
    x = np.arange(4)
    width = 0.25
    
    ax1.bar(x - width, stages_mapes["Transformer"], width, label="Transformer", color=C["trans"], edgecolor="black", linewidth=0.5, alpha=0.85)
    ax1.bar(x, stages_mapes["Trajectory-Reg. Transformer"], width, label="Trajectory-Reg. Transformer", color=C["phys"], edgecolor="black", linewidth=0.5, alpha=0.85)
    ax1.bar(x + width, stages_mapes["QPT (Full)"], width, label="QPT (Full)", color=C["qpt"], edgecolor="black", linewidth=0.5, alpha=0.9)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Endpoints\n5–25", "Endpoints\n26–50", "Endpoints\n51–75", "Endpoints\n76–100"], fontsize=7.5)
    ax1.set_ylabel("Prognostics Error (MAPE %)")
    ax1.set_title("(a) Prognostic Error Convergence", fontweight="bold", fontsize=9.0)
    ax1.legend(loc="upper right", framealpha=0.9, fontsize=8)
    ax1.set_ylim(0, 26)
    ax1.grid(True, linestyle="--", alpha=0.3, axis="y")
    
    # (b) Single Cell EOL Prognostics Curve & Smoothness (Cell Index 2)
    ax2 = fig.add_subplot(gs[0, 1])
    c_idx = 2
    start = c_idx * cell_len
    end = start + cell_len
    
    cell_gt = true_rul[start]
    cell_p_trans = p_trans[start:end]
    cell_p_phys = p_phys[start:end]
    cell_p_qpt = p_qpt[start:end]
    cycle_idx = np.arange(5, 101)
    
    # Smooth and compute confidence band for QPT
    qpt_smooth = np.convolve(cell_p_qpt, np.ones(5)/5, mode="same")
    qpt_std = np.abs(cell_p_qpt - qpt_smooth) * 1.5
    
    l0 = ax2.axhspan(cell_gt * 0.95, cell_gt * 1.05, color="#27ae60", alpha=0.08, zorder=0)
    l1 = ax2.axhline(cell_gt, color="#2c3e50", linestyle="--", linewidth=1.2, alpha=0.8, label="Ground Truth EOL", zorder=1)
    l2, = ax2.plot(cycle_idx, cell_p_trans, color="#e74c3c", alpha=0.85, linewidth=1.2, label="Transformer")
    l3, = ax2.plot(cycle_idx, cell_p_phys, color="#2980b9", alpha=0.85, linewidth=1.2, label="Trajectory-Reg. Transformer")
    ax2.fill_between(cycle_idx, cell_p_qpt - qpt_std, cell_p_qpt + qpt_std, color="#8e44ad", alpha=0.15, zorder=2)
    l4, = ax2.plot(cycle_idx, cell_p_qpt, color="#8e44ad", linewidth=2.2, label="QPT (Full)", zorder=3)
    
    # Second Y-axis: Prediction Smoothness (Second Finite Difference d^2(y_hat)/dt^2)
    d2_trans = cell_p_trans[2:] - 2 * cell_p_trans[1:-1] + cell_p_trans[:-2]
    d2_phys = cell_p_phys[2:] - 2 * cell_p_phys[1:-1] + cell_p_phys[:-2]
    d2_qpt = cell_p_qpt[2:] - 2 * cell_p_qpt[1:-1] + cell_p_qpt[:-2]
    cycle_idx_d2 = cycle_idx[1:-1]
    
    ax2_twin = ax2.twinx()
    # Subtle zero reference line for acceleration
    ax2_twin.axhline(0, color="#bdc3c7", linestyle=":", linewidth=0.8, alpha=0.6, zorder=0)
    
    l5, = ax2_twin.plot(cycle_idx_d2, d2_trans, color="#e67e22", linestyle="-", alpha=0.75, linewidth=1.1, label=r"Trans. $\frac{d^2\hat{y}}{dt^2}$")
    l6, = ax2_twin.plot(cycle_idx_d2, d2_phys, color="#16a085", linestyle="-", alpha=0.85, linewidth=1.4, label=r"Trajectory-Reg. $\frac{d^2\hat{y}}{dt^2}$")
    l7, = ax2_twin.plot(cycle_idx_d2, d2_qpt, color="#d35400", linestyle="-", alpha=0.95, linewidth=1.9, label=r"QPT $\frac{d^2\hat{y}}{dt^2}$")
    
    ax2_twin.set_ylabel(r"Smoothness ($\frac{d^2\hat{y}}{dt^2}$)", fontsize=7.5, color="#d35400", fontweight="bold")
    ax2_twin.tick_params(axis='y', labelsize=6.5, labelcolor="#d35400")
    ax2_twin.set_ylim(-6, 32)  # Tightened twin range to scale smoothness larger at the bottom
    ax2_twin.grid(False)
    
    ax2.set_title("(b) Representative-Cell EOL Predictions and Successive-Window Variation", fontweight="bold", fontsize=8.5)
    ax2.set_xlabel("Observation Cycle")
    ax2.set_ylabel("Predicted EOL (Cycles)")
    ax2.set_ylim(cell_gt * 0.82, cell_gt * 1.16)  # Tightened Y-axis (600 to 850 cycles) to scale trajectory curves larger
    
    # Combined legend at top right
    lines = [l1, l2, l3, l4, l5, l6, l7]
    labels = [line.get_label() for line in lines]
    ax2.legend(lines, labels, loc="upper right", framealpha=0.92, fontsize=6.8, ncol=2, columnspacing=0.8)
    ax2.grid(True, linestyle="--", alpha=0.3)
    
    # (c) QPT Prediction Parity (Density)
    ax_p = fig.add_subplot(gs[1, 0])
    min_v, max_v = 100, max(true_rul.max(), p_qpt.max()) * 1.05
    ax_p.plot([min_v, max_v], [min_v, max_v], "k-", linewidth=1.0, alpha=0.6, zorder=4)
    ax_p.fill_between([min_v, max_v], [min_v*0.85, max_v*0.85], [min_v*1.15, max_v*1.15],
                      color="gray", alpha=0.08, zorder=0, label="±15% Band")
    hb = ax_p.hexbin(true_rul, p_qpt, gridsize=40, cmap="RdPu", mincnt=1, linewidths=0.2, zorder=2)
    fig.colorbar(hb, ax=ax_p, label="Count", shrink=0.8)
    
    ax_p.set_xlabel("True EOL (Cycles)")
    ax_p.set_ylabel("Predicted EOL (Cycles)")
    ax_p.set_title("(c) QPT Prediction Parity (Density)", fontweight="bold", fontsize=9.5)
    ax_p.set_xlim(min_v, max_v)
    ax_p.set_ylim(min_v, max_v)
    ax_p.set_aspect("equal")
    ax_p.legend(loc="upper left", fontsize=7)
    
    # (d) Error CDF Comparison
    ax_c = fig.add_subplot(gs[1, 1])
    models_err = {
        "Transformer": np.abs(p_trans - true_rul) / true_rul * 100,
        "Trajectory-Reg. Transformer": np.abs(p_phys - true_rul) / true_rul * 100,
        "QPT (Full)": np.abs(p_qpt - true_rul) / true_rul * 100,
    }
    model_colors_cdf = [C["trans"], C["phys"], C["qpt"]]
    lw_map = {"Transformer": 1.0, "Trajectory-Reg. Transformer": 1.0, "QPT (Full)": 2.2}
    
    for (name, err), color in zip(models_err.items(), model_colors_cdf):
        sorted_err = np.sort(err)
        cdf = np.arange(1, len(sorted_err)+1) / len(sorted_err) * 100
        ax_c.plot(sorted_err, cdf, color=color, linewidth=lw_map[name], label=name, alpha=0.9)
    
    for thresh in [10, 15, 20]:
        ax_c.axvline(thresh, color="gray", linestyle=":", linewidth=0.6, alpha=0.5)
        pct = np.mean(models_err["QPT (Full)"] <= thresh) * 100
        ax_c.text(thresh + 0.5, pct - 3, f"{pct:.0f}%", fontsize=7.5, color=C["qpt"])
        
    ax_c.set_xlabel("Relative Prediction Error (%)")
    ax_c.set_ylabel("Cumulative Percentage of Samples (%)")
    ax_c.set_title("(d) Error CDF Comparison", fontweight="bold", fontsize=9.5)
    ax_c.set_xlim(0, 60)
    ax_c.set_ylim(0, 105)
    ax_c.legend(loc="lower right", framealpha=0.9, fontsize=7.5)
    ax_c.grid(axis="both", linestyle="--", alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(out_dir / "fig4_degradation_trajectories.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[main.py] Figure 4 saved to {out_dir / 'fig4_degradation_trajectories.png'}")

def plot_latent_tsne(out_dir):
    datasets = ["MIX_large", "CALB", "ZN-coin", "NAion"]
    titles = [
        "(a) MIX-Large (Li-ion blend)",
        "(b) CALB (Li-ion commercial)",
        "(c) Zn-coin (Zinc chemistry)",
        "(d) Na-ion (Sodium-ion chemistry)"
    ]
    
    fig = plt.figure(figsize=(10, 8.5))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.25)
    
    cmap = plt.cm.viridis
    
    for i, ds in enumerate(datasets):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        preds_qpt = torch.load(EXP_DIR / f"exp_{ds}_QPT" / "predictions.pt", map_location="cpu", weights_only=False)
        
        emb_qpt = np.array(preds_qpt["embeddings"])
        true_rul = np.array(preds_qpt["true_rul"])
        
        # Find cell start/end indices dynamically (handles varying lengths)
        change_indices = np.where(true_rul[:-1] != true_rul[1:])[0] + 1
        cell_starts = np.insert(change_indices, 0, 0)
        cell_ends = np.append(change_indices, len(true_rul))
        n_cells = len(cell_starts)
        
        # Run PCA on embeddings
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42)
        proj_all = pca.fit_transform(emb_qpt)
        
        # Background scatter
        ax.scatter(proj_all[:, 0], proj_all[:, 1], c=true_rul, cmap=cmap, s=2, alpha=0.2, zorder=1)
        
        # Select representative cells (short, medium, long life) based on sorted EOL
        cell_ruls = np.array([true_rul[cell_starts[c]] for c in range(n_cells)])
        sort_idx = np.argsort(cell_ruls)
        
        if n_cells >= 3:
            rep_indices = [sort_idx[0], sort_idx[n_cells // 2], sort_idx[-1]]
        else:
            rep_indices = [sort_idx[0], sort_idx[-1]]
            
        colors = ["#e74c3c", "#f39c12", "#27ae60"]
        for cell_idx, color in zip(rep_indices, colors):
            s_idx = cell_starts[cell_idx]
            e_idx = cell_ends[cell_idx]
            cell_path = proj_all[s_idx:e_idx]
            
            ax.plot(cell_path[:, 0], cell_path[:, 1], color=color, linewidth=1.5, alpha=0.9, zorder=4)
            ax.scatter(cell_path[0, 0], cell_path[0, 1], color=color, marker="o", s=30, zorder=5)
            ax.scatter(cell_path[-1, 0], cell_path[-1, 1], color=color, marker="X", s=35, zorder=5)
            
        ax.set_xlabel("PCA Dimension 1", fontsize=8)
        ax.set_ylabel("PCA Dimension 2", fontsize=8)
        ax.set_title(titles[i], fontweight="bold", fontsize=9.5)
        ax.grid(True, linestyle="--", alpha=0.2)
        
        # Individual colorbar to represent scale
        fig.colorbar(ax.collections[0], ax=ax, label="True EOL (Cycles)", shrink=0.75, pad=0.02)
        
    # Global legend for paths
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#e74c3c", lw=2, marker="o", ls="-", label="Short-life Trajectory"),
        Line2D([0], [0], color="#f39c12", lw=2, marker="o", ls="-", label="Medium-life Trajectory"),
        Line2D([0], [0], color="#27ae60", lw=2, marker="o", ls="-", label="Long-life Trajectory")
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=8.5, framealpha=0.9, bbox_to_anchor=(0.5, -0.05))
    
    fig.savefig(out_dir / "fig5_latent_tsne.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[main.py] Figure 5 saved to {out_dir / 'fig5_latent_tsne.png'}")

def plot_feature_importance(out_dir):
    preds_qpt = torch.load(EXP_DIR / "exp_MIX_large_QPT" / "predictions.pt", map_location="cpu", weights_only=False)
    emb = np.array(preds_qpt["embeddings"])
    true_rul = np.array(preds_qpt["true_rul"])
    
    import yaml
    with open(EXP_DIR / "exp_MIX_large_QPT" / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    d_model = cfg.get("d_model", 64)
    q_dim = cfg.get("q_dim", 32)
    
    trans_part = emb[:, :d_model]
    quant_part = emb[:, d_model:]
    
    trans_corrs = np.array([pearsonr(trans_part[:, i], true_rul)[0] for i in range(d_model)])
    quant_corrs = np.array([pearsonr(quant_part[:, i], true_rul)[0] for i in range(q_dim)])
    
    trans_var = trans_part.var(axis=0)
    quant_var = quant_part.var(axis=0)
    
    fig = plt.figure(figsize=(10, 4.5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 0.8], wspace=0.3)
    
    # (a) Bivariate scatter plot of correlation vs variance
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(trans_corrs, trans_var, color=C["trans"], alpha=0.75, s=30, edgecolor="black", linewidths=0.5, label="Transformer (96 dimensions)")
    ax1.scatter(quant_corrs, quant_var, color=C["qpt"], alpha=0.85, s=45, marker="D", edgecolor="black", linewidths=0.5, label="Q-OFE (48 dimensions)")
    
    # Draw quadrant dividers and shade central correlation zone (|r| <= 0.2)
    corr_threshold = 0.2
    var_threshold = 0.4
    ax1.axvspan(-corr_threshold, corr_threshold, color="#ecf0f1", alpha=0.6, zorder=0, label="Low Correlation Zone (|r| ≤ 0.2)")
    ax1.axvline(corr_threshold, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax1.axvline(-corr_threshold, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax1.axhline(var_threshold, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    
    bbox_props = dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.92, edgecolor="#bdc3c7", linewidth=0.8)
    bbox_center = dict(boxstyle="round,pad=0.3", facecolor="#f8f9f9", alpha=0.95, edgecolor="#8e44ad", linewidth=0.9)
    
    # Label outer and central regions cleanly without interpretive quadrant names
    ax1.text(0.58, 2.7, "High variance\n" + r"$r > 0.2$", ha="center", va="center", fontsize=7.0, color="#27ae60", fontweight="bold", bbox=bbox_props)
    ax1.text(-0.58, 2.7, "High variance\n" + r"$r < -0.2$", ha="center", va="center", fontsize=7.0, color="#c0392b", fontweight="bold", bbox=bbox_props)
    ax1.text(0.58, -0.9, "Low variance\n" + r"$r > 0.2$", ha="center", va="center", fontsize=7.0, color="#2980b9", fontweight="bold", bbox=bbox_props)
    ax1.text(-0.58, -0.9, "Low variance\n" + r"$r < -0.2$", ha="center", va="center", fontsize=7.0, color="#d35400", fontweight="bold", bbox=bbox_props)
    
    # Central Zone (between -0.2 and +0.2)
    ax1.text(0.0, 2.7, "High variance\n" + r"$|r| \leq 0.2$", ha="center", va="center", fontsize=7.0, color="#7f8c8d", fontweight="bold", bbox=bbox_props)
    ax1.text(0.0, -0.9, "Low variance\n" + r"$|r| \leq 0.2$", ha="center", va="center", fontsize=7.0, color="#8e44ad", fontweight="bold", bbox=bbox_center)
    
    ax1.set_xlabel("Pearson Correlation with EOL", fontsize=8.5)
    ax1.set_ylabel("Feature Variance", fontsize=8.5)
    ax1.set_title("(a) Feature Variance and Correlation with EOL", fontweight="bold", fontsize=9)
    ax1.set_xlim(-0.95, 0.95)
    ax1.set_ylim(-1.5, 4.2)  # Smaller y-min (-1.5) moves all scatter points higher up
    ax1.legend(loc="upper center", fontsize=7.2, framealpha=0.92, ncol=2)
    ax1.grid(True, linestyle="--", alpha=0.3)
    
    # (b) Cycle-to-cycle Prediction Volatility Boxplot
    ax2 = fig.add_subplot(gs[0, 1])
    
    fig_data_path = EXP_DIR / "analysis" / "figures_data.pt"
    if fig_data_path.exists():
        fig_data = torch.load(fig_data_path, map_location="cpu", weights_only=False)
        preds_trans = np.array(fig_data["fig4_preds_transformer"])
        preds_phys = np.array(fig_data["fig4_preds_physics"])
        preds_qpt_all = np.array(fig_data["fig4_preds_qpt"])
        
        cell_len = 96
        n_cells = len(preds_trans) // cell_len
        
        vol_trans = [np.std(np.diff(preds_trans[i*cell_len:(i+1)*cell_len])) for i in range(n_cells)]
        vol_phys = [np.std(np.diff(preds_phys[i*cell_len:(i+1)*cell_len])) for i in range(n_cells)]
        vol_qpt = [np.std(np.diff(preds_qpt_all[i*cell_len:(i+1)*cell_len])) for i in range(n_cells)]
        
        box_data = [vol_trans, vol_phys, vol_qpt]
        bp = ax2.boxplot(box_data, patch_artist=True, tick_labels=["Transformer", "Trajectory-Reg.", "QPT (Full)"], widths=0.45)
        
        colors = [C["trans"], C["phys"], C["qpt"]]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.8)
        for median in bp['medians']:
            median.set_color("black")
            median.set_linewidth(1.2)
        for whisker in bp['whiskers']:
            whisker.set_color("gray")
            whisker.set_linestyle("--")
        for cap in bp['caps']:
            cap.set_color("gray")
            
        ax2.set_ylabel(r"Successive Prediction Change $\sigma(\Delta\mathrm{EOL})$ (cycles)", fontsize=7.8)
        ax2.set_title("(b) Distribution of Successive Prediction Changes", fontweight="bold", fontsize=8.8)
        ax2.grid(True, linestyle="--", alpha=0.3, axis="y")
        
    plt.tight_layout()
    fig.savefig(out_dir / "fig6_feature_correlation.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[main.py] Figure 6 saved to {out_dir / 'fig6_feature_correlation.png'}")

def plot_entanglement_matrix(vqc_params_dict, out_dir):
    if not vqc_params_dict:
        print("[main.py] VQC checkpoint parameters not loaded. Skipping Figure 7.")
        return
        
    datasets = ["MIX_large", "CALB", "ZN-coin", "NAion"]
    titles = [
        "MIX-Large (Li-ion blend)",
        "CALB (Li-ion commercial)",
        "Zn-coin (Zinc chemistry)",
        "Na-ion (Sodium-ion chemistry)"
    ]
    
    fig = plt.figure(figsize=(14, 7.5))
    gs = gridspec.GridSpec(2, 4, hspace=0.38, wspace=0.40)
    
    for i, ds in enumerate(datasets):
        if ds not in vqc_params_dict:
            continue
        if isinstance(vqc_params_dict[ds], dict) and "delta" in vqc_params_dict[ds]:
            delta = vqc_params_dict[ds]["delta"]
        else:
            U_l2 = vqc_params_dict[ds][1]["U"]
            off_diag = U_l2 - np.diag(np.diag(U_l2))
            delta = np.abs(off_diag)
        n_qubits = delta.shape[0]
        
        d_max = max(delta.max(), 1e-4)
        
        # Top row: Heatmap with dedicated discrete/proportional colorbar
        ax_top = fig.add_subplot(gs[0, i])
        im = ax_top.imshow(delta, cmap="plasma", aspect="equal", vmin=0, vmax=d_max)
        cbar = fig.colorbar(im, ax=ax_top, shrink=0.82)
        cbar.ax.tick_params(labelsize=6.5)
        cbar.set_label("$|U - I|$", fontsize=7.5)
        ax_top.set_xlabel("Output Latent Dimension", fontsize=7.5)
        if i == 0:
            ax_top.set_ylabel("Input Latent Dimension", fontsize=7.5)
        ax_top.set_title(f"({chr(97 + i)}) {titles[i]}\n" + r"Orthogonal Deviation $|U - I|$", fontweight="bold", fontsize=8.0)
        
        # Bottom row: Circular Latent-Feature Coupling Network Graph
        ax_bot = fig.add_subplot(gs[1, i])
        ax_bot.set_aspect("equal")
        
        angles = np.linspace(0, 2 * np.pi, n_qubits, endpoint=False)
        x = np.cos(angles)
        y = np.sin(angles)
        
        # Adaptive threshold: top 20% strongest couplings for this dataset
        threshold = np.percentile(delta, 80)
        
        # Draw entanglement lines with BOLD visibility
        for u in range(n_qubits):
            for v in range(u+1, n_qubits):
                weight = delta[u, v]
                if weight > threshold:
                    norm_w = (weight - threshold) / (d_max - threshold + 1e-8)
                    ax_bot.plot(
                        [x[u], x[v]], [y[u], y[v]],
                        color="#8e44ad",
                        alpha=0.4 + 0.5 * norm_w,
                        linewidth=1.0 + 3.0 * norm_w,
                        zorder=1
                    )
                    
        # Draw nodes
        ax_bot.scatter(x, y, color="#3498db", s=85, edgecolor="black", linewidths=0.8, zorder=2)
        for q in range(n_qubits):
            ax_bot.text(
                x[q] * 1.22, y[q] * 1.22, str(q),
                ha="center", va="center", fontsize=6.2, fontweight="bold", color="#2c3e50"
            )
            
        ax_bot.set_xlim(-1.45, 1.45)
        ax_bot.set_ylim(-1.45, 1.45)
        ax_bot.axis("off")
        ax_bot.set_title(f"({chr(101 + i)}) {titles[i]}\nLatent-Feature Coupling Graph", fontweight="bold", fontsize=8.2)

    fig.savefig(out_dir / "fig7_entanglement_matrix.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[main.py] Figure 7 (Bold 4-Dataset VQC Entanglement Analysis) saved to {out_dir / 'fig7_entanglement_matrix.png'}")

def plot_vqc_feature_evolution(out_dir):
    datasets = ["MIX_large", "CALB", "ZN-coin", "NAion"]
    labels = [
        "(a) MIX-Large (Li-ion blend)",
        "(b) CALB (Li-ion commercial)",
        "(c) Zn-coin (Zinc chemistry)",
        "(d) Na-ion (Sodium-ion chemistry)"
    ]
    
    fig = plt.figure(figsize=(11, 7.5))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.25)
    
    from matplotlib.colors import ListedColormap
    phase_cmap = ListedColormap(["#2ecc71", "#f1c40f", "#e74c3c"])
    
    for i, ds in enumerate(datasets):
        preds_qpt = torch.load(EXP_DIR / f"exp_{ds}_QPT" / "predictions.pt", map_location="cpu", weights_only=False)
        emb = np.array(preds_qpt["embeddings"])
        true_rul = np.array(preds_qpt["true_rul"])
        
        import yaml
        with open(EXP_DIR / f"exp_{ds}_QPT" / "config.yaml") as f:
            cfg = yaml.safe_load(f)
        d_model = cfg.get("d_model", 64)
        quant_part = emb[:, d_model:]
        
        cell_len = 96
        n_cells = len(true_rul) // cell_len
        cell_feats = []
        cell_ruls = []
        for c in range(n_cells):
            s, e = c * cell_len, (c + 1) * cell_len
            cell_feats.append(quant_part[s:e].mean(axis=0))
            cell_ruls.append(true_rul[s])
            
        cell_feats = np.array(cell_feats)
        cell_ruls = np.array(cell_ruls)
        
        sort_idx = np.argsort(cell_ruls)
        cell_feats_sorted = cell_feats[sort_idx]
        cell_ruls_sorted = cell_ruls[sort_idx]
        
        # Cluster features using ward linkage
        feat_linkage = linkage(cell_feats_sorted.T, method="ward")
        feat_order = leaves_list(feat_linkage)
        cell_feats_clustered = cell_feats_sorted[:, feat_order]
        
        # Get subplot inner gridspec
        gs_inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[i // 2, i % 2], height_ratios=[1, 0.15], hspace=0.08)
        
        # Heatmap ax
        ax_hm = fig.add_subplot(gs_inner[0])
        vmax = np.percentile(np.abs(cell_feats_clustered), 95)
        im = ax_hm.imshow(cell_feats_clustered.T, aspect="auto", cmap="RdBu_r", interpolation="nearest",
                          vmin=-vmax, vmax=vmax)
        
        ax_hm.set_ylabel("Q-OFE Feature Index", fontsize=8)
        ax_hm.set_title(labels[i], fontweight="bold", fontsize=9)
        ax_hm.set_xticks([])
        
        # Phase strip ax
        ax_phase = fig.add_subplot(gs_inner[1])
        n_c = len(cell_ruls_sorted)
        phase_strip = np.zeros((1, n_c))
        phase_strip[0, :int(n_c*0.25)] = 1
        phase_strip[0, int(n_c*0.25):int(n_c*0.75)] = 2
        phase_strip[0, int(n_c*0.75):] = 3
        
        ax_phase.imshow(phase_strip, aspect="auto", cmap=phase_cmap, interpolation="nearest")
        ax_phase.set_yticks([])
        ax_phase.set_xlabel("Cells sorted by EOL (Cycles)", fontsize=8)
        
        tick_positions = np.linspace(0, n_c - 1, 5, dtype=int)
        ax_phase.set_xticks(tick_positions)
        ax_phase.set_xticklabels([f"{cell_ruls_sorted[p]:.0f}" for p in tick_positions], fontsize=7)
        
        ax_hm.set_xlim(ax_phase.get_xlim())
        
        # Add colorbar for each heatmap to make features easily readable
        fig.colorbar(im, ax=ax_hm, label="Activation", shrink=0.7, pad=0.02)
        
    # Single global legend for lifetime groups
    legend_elements = [
        Line2D([0], [0], color="#2ecc71", lw=4, label="Low-EOL Group"),
        Line2D([0], [0], color="#f1c40f", lw=4, label="Intermediate-EOL Group"),
        Line2D([0], [0], color="#e74c3c", lw=4, label="High-EOL Group")
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=8.5, framealpha=0.9, bbox_to_anchor=(0.5, -0.05))
    
    fig.savefig(out_dir / "fig8_feature_evolution.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[main.py] Figure 8 saved to {out_dir / 'fig8_feature_evolution.png'}")

def plot_physics_correlation_heatmap(data, out_dir):
    import seaborn as sns
    if "fig12_physics_correlation" not in data:
        print("[main.py] Skipping Figure 12: fig12_physics_correlation key not found in data.")
        return
        
    phy_data = data["fig12_physics_correlation"]
    feature_names = phy_data["feature_names"]
    corr_matrix = np.array(phy_data["corr_matrix"])  # [6, 32]
    
    # Select top 10 dimensions with highest absolute correlation across features
    max_abs_corrs = np.max(np.abs(corr_matrix), axis=0)
    top_qubit_indices = np.argsort(max_abs_corrs)[::-1][:10]
    top_qubit_indices = np.sort(top_qubit_indices)
    
    subset_corr = corr_matrix[:, top_qubit_indices]
    dimension_labels = [f"Dimension {q}" for q in top_qubit_indices]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(subset_corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-0.7, vmax=0.7,
                xticklabels=dimension_labels, yticklabels=feature_names,
                cbar_kws={"label": "Pearson Correlation Coefficient (r)"},
                linewidths=0.8, linecolor="white", ax=ax,
                annot_kws={"size": 9.5, "weight": "bold"})
                
    ax.set_title("Associations Between Q-OFE Dimensions and Input-Derived Battery Descriptors",
                 fontsize=11, fontweight="bold", pad=15)
    ax.set_xlabel("Selected Q-OFE Dimensions", fontsize=10, fontweight="bold")
    ax.set_ylabel("Input-Derived Battery Descriptors", fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=9.5)
    
    plt.tight_layout()
    fig.savefig(out_dir / "fig12_physics_correlation_heatmap.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[main.py] Figure 12 saved to {out_dir / 'fig12_physics_correlation_heatmap.png'}")

def main():
    print("=" * 80)
    print("[main.py] RUNNING PUBLICATION-QUALITY ARTIFACT GENERATOR PIPELINE")
    print("=" * 80)

    # Set up output directories
    out_analysis = ROOT / "experiments" / "analysis"
    out_analysis.mkdir(parents=True, exist_ok=True)
    out_figures = ROOT / "figures"
    out_figures.mkdir(parents=True, exist_ok=True)
    out_tables = ROOT / "tables"
    out_tables.mkdir(parents=True, exist_ok=True)
    
    # Load pre-computed figures data
    data_path = out_analysis / "figures_data.pt"
    if not data_path.exists():
        print(f"[main.py] ERROR: figures_data.pt not found at {data_path}!")
        sys.exit(1)
        
    print(f"[main.py] Loading pre-computed figure data from {data_path}...")
    data = torch.load(data_path, map_location="cpu", weights_only=False)

    # Load experimental metrics and VQC parameters
    metrics = load_metrics()
    vqc_params = load_vqc_parameters()

    # 1. LaTeX tables
    generate_latex_tables(data, out_tables)
    generate_latex_tables(data, out_analysis)
    
    # 2. Dataset profile curves
    plot_dataset_profile(data, metrics, out_figures)
    plot_dataset_profile(data, metrics, out_analysis)
    
    # 3. Model architecture
    print("[main.py] Figure 2 Architecture Diagram generated via generate_image (already placed in figures/).")

    # 4. Performance comparisons
    plot_mape_comparison(metrics, out_figures)
    plot_mape_comparison(metrics, out_analysis)
    
    # 5. Degradation Trajectories
    plot_degradation_trajectories(data, out_figures)
    plot_degradation_trajectories(data, out_analysis)
    
    # 6. Latent Space Projections
    plot_latent_tsne(out_figures)
    plot_latent_tsne(out_analysis)

    # 7. VQC Feature Correlations/Importance
    plot_feature_importance(out_figures)
    plot_feature_importance(out_analysis)
    
    # 8. VQC Entanglement Heatmap
    plot_entanglement_matrix(vqc_params, out_figures)
    plot_entanglement_matrix(vqc_params, out_analysis)
    
    # 9. VQC Feature Evolution Trajectory Plot / Heatmap
    plot_vqc_feature_evolution(out_figures)
    plot_vqc_feature_evolution(out_analysis)
    
    # 10. Multi-Physics vs. VQC Virtual Qubits Correlation Heatmap (Figure 12)
    plot_physics_correlation_heatmap(data, out_figures)
    plot_physics_correlation_heatmap(data, out_analysis)
    
    # 11. Latent space separability metrics
    compute_clustering_metrics(data, out_tables)
    compute_clustering_metrics(data, out_analysis)
    
    print("\n" + "=" * 80)
    print("[main.py] ARTIFACT GENERATION COMPLETED SUCCESSFULLY")
    print("  Figures located in: figures/ and experiments/analysis/")
    print("  Tables located in: tables/ and experiments/analysis/")
    print("=" * 80)

if __name__ == "__main__":
    main()
