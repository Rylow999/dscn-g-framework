# DSCN-G: Dual-State Cognitive Geometry

**Author:** Luciano Benjamín Nieto  
**Contact:** lucianobenjaminnieto@gmail.com  
**License:** MIT

---

## Overview

DSCN-G (Dual-State Cognitive Geometry) is a unified computational architecture that models cognition as an emergent property of autopoietic hierarchical graphs. The system integrates:

- High-dimensional state vectors evolved via stochastic TD-learning
- Bounded Kuramoto phase dynamics
- *K* parallel information chains with probabilistic transitions
- Activity-dependent structural plasticity
- O(log *N*) memory recovery via harmonic resonance

Three formal theorems are computationally verified over 100 independent seeds × 2000 steps (200,000 total state evaluations):

- **Theorem 1** — Homeostatic stability: *N* ≤ ρ_eff/θ_death ≈ 7.00
- **Theorem 2** — Vector attractors in O(β)-neighborhood: ω* = 0.649 ± 0.10
- **Theorem 3** — Phase convergence: *p*_conv = 0.97 > 0.5

The **C3 Prediction** (Phase-Hijacking of valence) constitutes the framework's primary differentiating contribution: a falsifiable, directional, thresholded, and causally directed phase perturbation testable with EEG γ-band in acute pain paradigms.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run complete reproduction suite (~3 minutes on standard CPU)
python src/dscn_g_v7_2.py all

# Verify data integrity against paper claims
python src/verify_submission.py
```

---

## Structure

```text
.
├── README.md                    # This file
├── LICENSE                      # MIT License
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
├── CITATION.cff                 # Citation metadata
├── src/
│   ├── dscn_g_v7_2.py           # Complete simulator (all experiments)
│   └── verify_submission.py     # Automated verification script
├── data/
│   ├── base_simulation_100seeds.json
│   ├── scalability_results.json
│   ├── parameter_sweep.json
│   ├── field_results.json
│   └── quantum_results.json
├── figures/
│   ├── fig1_base_simulation.png
│   ├── fig2_scalability.png
│   ├── fig3_hijacking_sweep.png
│   └── fig4_autopoietic_field.png
└── paper/
    └── dscn_g_paper.md          # Markdown version (GitHub-ready)
```

---

## Reproducibility

All simulations use deterministic RNG (`numpy.random.default_rng(seed)`). The complete suite regenerates all data, figures, and verification reports.

**Expected outputs:**

- `data/*.json` — Verification data
- `figures/*.png` — All 4 figures
- Console report with all metrics

---

## Key Results

| Metric | Simulation | Theoretical | Status |
|--------|-----------|-------------|--------|
| *N* | 4.00 ± 0.00 | ≤ 7.00 | ✓ T.1 verified |
| ω* | 0.612 ± 0.173 | 0.649 ± 0.10 | ✓ T.2 verified |
| *p*_conv | 0.97 | > 0.50 | ✓ T.3 verified |
| Hijack rate | 28.6% | — | Prediction C3 |

---

## Citation

```bibtex
@software{Nieto2026DSCNG,
  title={DSCN-G: Dual-State Cognitive Geometry},
  author={Nieto, Luciano Benjamín},
  year={2026},
  url={https://github.com/Rylow999/dscn-g-framework}
}
```

---

## Related Work

See the companion repository **[dscn-g-bio](https://github.com/Rylow999/dscn-g-bio)** for the biological testability framework, EEG predictions, and experimental protocols derived from DSCN-G.