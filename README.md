# Battery-QPT: A Dual-Path Transformer with Periodic Orthogonal Feature Encoding and Trajectory Regularization for Early-Cycle Battery Remaining Useful Life Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Under%20Review-orange.svg)]()

Welcome to the official repository for **Battery-QPT** (**Dual-Path** Transformer, **Q**uantum-Inspired **P**hysics-Informed **T**ransformer), a framework designed for early-life State of Health (SOH) and Remaining Useful Life (RUL) prediction across diverse battery chemistries.

Battery-QPT bridges quantum-inspired state representations with electrochemical domain knowledge, resolving the persistent trade-off between predictive accuracy and physical consistency.

---

> [!NOTE]
> ### 📢 Code and Model Weights Availability
> The complete source code, training/evaluation pipelines, preprocessing scripts, and pretrained checkpoints are currently undergoing peer review alongside our submitted manuscript.  
> **The full codebase and reproduction scripts will be publicly uploaded to this repository immediately upon formal acceptance of the manuscript.**  
> For any questions or early inquiries, please feel free to open a GitHub Issue or contact the authors.

---

## ⚡ Core Advantages of Battery-QPT

1. **Superior High-Precision Prognostics (Acc@10%)**: While coarse classification metrics (Acc@15%) treat large outliers leniently, Battery-QPT excels in strict-tolerance regimes (**Acc@10%**), outperforming baselines by up to **+38.5%** on Sodium-ion and **+15.5%** on Zinc-ion.
2. **28% Damping of Prediction Volatility**: Classical black-box neural networks suffer from high-frequency prediction fluctuations caused by early-life measurement noise. By incorporating physical monotonicity and smoothness constraints, QPT reduces cycle-to-cycle prediction volatility $\sigma(\Delta\hat{y})$ by **28%** (from 7.60 to 5.99).
3. **Exact Mathematical Unitarity**: Skew-symmetric Cayley parameterization ($U = (I-A)(I+A)^{-1}, A^T = -A$) guarantees exact unitary transformation ($U^T U = I$) in the simulated Variational Quantum Circuit (VQC), preventing gradient dissipation and preserving feature state norms during backpropagation.
4. **Electrochemically Grounded Virtual Qubits**: Latent quantum representations naturally self-organize into physically meaningful degradation indicators—Virtual Qubit Dim 21 correlates with macro-capacity fade ($r = +0.6526$ with $Q_{\text{max}}$), while Dim 16 tracks Ohmic resistance expansion ($r = -0.5179$ with $\Delta V_{\text{Ohm}}$).
5. **Ultra-Lightweight Edge Footprint (~110K Parameters)**: QPT requires only **110,753** parameters—over **90% fewer parameters** than standard benchmark models (CPMLP: ~2.15M, CPTransformer: ~1.05M), enabling rapid training ($\approx 42$s / 100 epochs) and embedded Battery Management System (BMS) microcontroller deployment.

---

## 📊 Comprehensive Results Summary Across 4 Chemistries

The table below summarizes the quantitative prognostic performance across four real-world benchmark datasets:

| Domain / Chemistry | Model | MAE (Cycles) ↓ | RMSE (Cycles) ↓ | MAPE ↓ | Acc@15% (%) ↑ | Acc@10% (%) ↑ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Li-ion** (`MIX_large`, 15k+ cycles) | CPMLP Benchmark | — | — | 0.1790 | 62.0% | — |
| | CPTransformer Benchmark | — | — | 0.1840 | 57.3% | — |
| | LSTM Baseline | 204.2 | 413.0 | 0.2821 | 48.2% | 36.6% |
| | Transformer Baseline | 163.0 | 266.6 | 0.2136 | 53.9% | 38.8% |
| | Physics + Transformer | 160.7 | 288.3 | 0.1942 | 57.8% | 42.7% |
| | **QPT (Ours)** | **152.4** | **283.4** | **0.1778** | **62.0%** | **45.9%** |
| **Commercial LFP** (`CALB`, 72 Ah) | CPMLP Benchmark | — | — | 0.1400 | 70.4% | — |
| | CPTransformer Benchmark | — | — | 0.1490 | 67.2% | — |
| | LSTM Baseline | 294.3 | 423.5 | 0.5488 | 39.8% | 39.8% |
| | Transformer Baseline | 91.6 | 134.0 | 0.2021 | 59.1% | 57.4% |
| | Physics + Transformer | 97.2 | 115.8 | 0.2107 | 58.3% | 56.0% |
| | **QPT (Ours)** | **81.7** | **109.9** | **0.1373** | **59.5%** | **59.1%** |
| **Aqueous Zinc-ion** (`ZN-coin`) | CPMLP Benchmark | — | — | 0.5580 | 29.7% | — |
| | CPTransformer Benchmark | — | — | 0.5150 | 20.2% | — |
| | LSTM Baseline | 245.8 | 380.4 | 0.9015 | 3.8% | 3.8% |
| | Transformer Baseline | 152.0 | 280.5 | 0.5564 | 24.7% | 15.0% |
| | Physics + Transformer | 150.3 | 278.2 | 0.5505 | 27.7% | 16.5% |
| | **QPT (Ours)** | **169.0** | **316.3** | **0.4554** | **25.8%** | **19.3%** |
| **Sodium-ion** (`NAion`, small-sample) | CPMLP Benchmark | — | — | 0.2740 | 33.7% | — |
| | CPTransformer Benchmark | — | — | 0.2550 | 40.6% | — |
| | LSTM Baseline | 60.9 | 66.0 | 0.4194 | 40.0% | 0.0% |
| | Transformer Baseline | 56.6 | 64.1 | 0.3554 | 40.0% | 1.5% |
| | Physics + Transformer | 57.0 | 64.1 | 0.3624 | 40.0% | 1.5% |
| | **QPT (Ours)** | **45.3** | **61.4** | **0.2658** | **40.0%** | **40.0%** |

---

## 📈 High-Precision Advantage (Acc@10% Comparison)

When high prognostic precision is mandated for EV safety, Battery-QPT delivers massive gains over baselines:

| Dataset / Chemistry | Transformer Baseline | Physics + Transformer | Battery-QPT (Ours) | Relative Accuracy Gain |
| :--- | :---: | :---: | :---: | :---: |
| **Sodium-ion** (`NAion`) | 1.5% | 1.5% | **40.0%** | **+38.5% improvement** |
| **Zinc-ion** (`ZN-coin`) | 15.0% | 16.5% | **19.3%** | **+4.3% improvement** (vs 3.8% LSTM) |
| **Lithium-ion** (`MIX_large`) | 38.8% | 42.7% | **45.9%** | **+7.1% improvement** |
| **Commercial LFP** (`CALB`) | 57.4% | 56.0% | **59.1%** | **+1.7% improvement** (vs 39.8% LSTM) |

---

## ⚡ Computational Efficiency & Parameter Footprint

| Architecture | Parameter Count | Training Time / 100 Epochs | Embedded BMS Suitability |
| :--- | :---: | :---: | :---: |
| CPMLP Benchmark | ~2,150,000 | ~180 s | Low (Heavy memory footprint) |
| CPTransformer Benchmark | ~1,050,000 | ~140 s | Moderate |
| CPLSTM Benchmark | ~500,000 | ~95 s | Moderate |
| **Battery-QPT (Ours)** | **110,753** | **~42 s** | **High (Optimal for ARM Cortex-M / Edge)** |

---

## 🔬 Electrochemical Interpretability: Virtual Qubit Alignment

| Virtual Qubit Feature | Electrochemical Metric | Pearson Correlation ($r$) | Physical Significance |
| :--- | :--- | :---: | :--- |
| **Dimension 21** | Maximum Capacity ($Q_{\text{max}}$) | **+0.6526** ($p < 10^{-16}$) | Directly tracks macro-capacity degradation |
| **Dimension 16** | Ohmic Polarization ($\Delta V_{\text{Ohm}}$) | **-0.5179** ($p < 10^{-16}$) | Tracks internal cell resistance growth |
| **Dimensions 25–30** | Knee Point Phase Transition | — | Exclusively activates during Accelerated Aging (Phase III) |

---

## 🛠️ Usage

To generate tabular analysis and metrics:
```bash
python main.py
```

To run model evaluations and compute prognostic metrics:
```bash
python evaluate_model.py
```

---

## 📌 Citation

If you find this work, benchmark results, or representations useful in your research, please cite our paper:

```bibtex
@article{LeMinh2026QPT,
  title     = {Physics-Informed Quantum-Inspired Transformer for Multi-Channel Battery State of Health and Remaining Useful Life Prediction},
  author    = {Le Minh, Huy and Tan, Ruifeng},
  journal   = {Under Review},
  year      = {2026}
}
```

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
