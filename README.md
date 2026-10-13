# Battery-QPT: Physics-Informed Quantum-Inspired Transformer for Battery State of Health and Remaining Useful Life Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Under%20Review-orange.svg)]()

Welcome to the official repository for **Battery-QPT** (**Q**uantum-Inspired **P**hysics-Informed **T**ransformer), a framework designed for early-life State of Health (SOH) and Remaining Useful Life (RUL) prediction across diverse battery chemistries.

Battery-QPT bridges quantum-inspired state representations with electrochemical domain knowledge, resolving the trade-off between predictive accuracy and physical consistency.

---

> [!NOTE]
> ### 📢 Code and Model Weights Availability
> The complete source code, training/evaluation pipelines, preprocessing scripts, and pretrained checkpoints are currently undergoing peer review alongside our submitted manuscript.  
> **The full codebase and reproduction scripts will be publicly uploaded to this repository immediately upon formal acceptance of the manuscript.**  
> For any questions or early inquiries, please feel free to open a GitHub Issue or contact the authors.

---

## ⚡ Highlights

* **Cross-Chemistry Generalizability**: Evaluated across 4 distinct battery platforms—**Li-ion** (`MIX_large`), **Commercial LFP** (`CALB`), **Aqueous Zn-ion** (`ZN-coin`), and **Sodium-ion** (`NAion`).
* **Quantum Hilbert Feature Mapping**: Maps raw multi-channel charging/discharging curves (voltage, current, capacity) into high-dimensional quantum states using simulated Variational Quantum Circuits (VQC) with skew-symmetric Cayley unitary transformations ($U^T U = I$), guaranteeing exact norm preservation and eliminating gradient dissipation.
* **Physics-Informed Loss Constraints**: Directly embeds electrochemical degradation laws into the training objective:
  * **Monotonicity Loss ($\mathcal{L}_{\text{mono}}$)**: Prevents unphysical capacity recovery.
  * **Smoothness Loss ($\mathcal{L}_{\text{smooth}}$)**: Suppresses erratic, cycle-to-cycle prediction volatility.
  * **Capacity Consistency Loss ($\mathcal{L}_{\text{capacity}}$)**: Enforces physical bounds derived from thermodynamic limits.
* **Continuous Degradation Manifolds**: Maps battery health along smooth thermodynamic decay manifolds, eliminating the artificial clustering and noise-fitting common to standard unconstrained deep learning models.

---

## 📊 Results Summary

The table below compares **Battery-QPT** against the domain winners from the ACM KDD 2025 *BatteryLife* benchmark across all 4 battery domains:

| Domain / Chemistry | Best Paper Baseline | Paper Best MAPE | Battery-QPT MAPE ↓ | Battery-QPT Acc@15% ↑ | Battery-QPT Acc@10% ↑ |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Li-ion** (`MIX_large`) | CPMLP | 0.179 | **0.1778** | **62.0%** | **45.9%** |
| **Commercial LFP** (`CALB`) | CPMLP | 0.140 | **0.1373** | **59.5%** | **59.1%** |
| **Zn-ion** (`ZN-coin`) | CPTransformer | 0.515 | **0.4554** | **25.8%** | **19.3%** |
| **Na-ion** (`NAion`) | CPTransformer | 0.255 | **0.2658** | **40.0%** | **40.0%** |

> [!TIP]
> **Battery-QPT achieves lower MAPE across heterogeneous Li-ion, commercial LFP, and Zn-ion batteries, with a substantial 49.5% relative error reduction over sequential LSTMs on dendrite-prone Zn-ion coin cells.**

A detailed breakdown of all validation scores, ablation studies, and physical consistency metrics is available in **[results.md](results.md)**.

---

## 🏛️ Method Overview

1. **Intra-Cycle Feature Re-Uploading**: Raw voltage, current, and capacity sequences are encoded via continuous harmonic mappings onto virtual Bloch sphere rotations.
2. **Norm-Preserving Entanglement**: Multi-qubit interactions are mediated through learnable skew-symmetric Cayley parameterizations:
   $$U = (I - A)(I + A)^{-1}, \quad A^T = -A \implies U^T U = I$$
   ensuring strict unitarity without numerical state dissipation.
3. **Temporal Attention Backbone**: Aggregates early-life cycle sequences to track long-term capacity fade trends.
4. **Thermodynamic Loss Bounds**: Combines prediction error with physical monotonicity, trajectory smoothness, and boundary constraints.

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
