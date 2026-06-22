# DDSD Part 3: Thermodynamic Confinement in Discrete Dynamical Systems

**Author:** Luciano Benjamín Nieto  
**Affiliation:** Independent Researcher, General Alvear, Mendoza, Argentina  
**Contact:** <lucianobenjaminnieto@gmail.com>  
**License:** MIT

---

## Overview

DDSD Part 3 presents computational evidence for a **spectral phase transition** in the family of accelerated Collatz maps $R_a(n) = (an+1)/2^{\nu_2(an+1)} \bmod 2^K$, parameterized by the odd coefficient $a$.

The framework identifies three distinct dynamical types:

- **Type I** (Collatz, $a=3$) — Clean spectral gap ($\lambda = 3/4$, drift $= -0.415$) with convergence to $n=1$ up to $K=30$
- **Type II** (Neutral, $a=5$) — Marginal spectrum ($\lambda = 1.0$) dominating 94.9% of dynamics at $K=28$
- **Type III** (Ultra-Champion, $a=7$) — Multiple unstable cycles (eigenvalues up to 3.5) dominating 99.997% of dynamics at $K=30$

All results are cross-verified with three independent implementations (Python dict, NumPy, C with path compression) and computationally verified up to $K=30$.

---

## Quick Start

```bash
# Install dependencies
pip install numpy matplotlib tqdm

# Run Collatz verification for K=13..20
python scripts/run_collatz_sweep.py --a 3 --kmin 10 --kmax 20

# Generate phase diagram for a=3,5,7,9,11 at K=20
python scripts/run_phase_diagram.py --k 20

# Generate all figures
python src/python/plots.py
```

**Performance (C Implementation):**

```bash
cd src/c
gcc -O3 -march=native -ffast-math -o fate_v3 fate_v3.c -lm

# Run Collatz at K=20
./fate_v3 20 3

# Run Ultra-Champion at K=30
./fate_v3 30 7
```

---

## Structure

```
thermodynamic-confinement-ddsd/
├── README.md                    # This file
├── LICENSE                      # MIT License
├── paper/
│   └── thermodynamic_confinement_v4.md
├── src/
│   ├── c/
│   │   ├── fate_v2.c            # 5 bytes/node implementation
│   │   └── fate_v3.c            # 3 bytes/node (K=30 optimized)
│   └── python/
│       ├── fate.py              # Reference implementation
│       ├── verify_cycles.py     # Cross-verification suite
│       └── plots.py             # Figure generation
├── scripts/
│   ├── run_collatz_sweep.py     # Tables 4.1–4.5 (a=3, K=10..30)
│   ├── run_phase_diagram.py     # Phase diagram (Section 7)
│   ├── run_uc_analysis.py       # Ultra-Champion analysis (Section 6)
│   └── run_all_verifications.py # Master verification
├── data/
│   └── *.json / *.csv           # Computational results
└── figures/
    └── *.png                    # Generated visualizations
```

---

## Reproducibility

All results use deterministic, seeded cycle enumeration. The master verification script reproduces all tables and figures from the paper in ~15 minutes on a standard CPU.

**Expected outputs:**

- K=13 transition detection for $a=3$ (3 cycles → 1 cycle)
- K=20 resonance (L=22, 3-block structure)
- Neutral cycles for $a=5$ at $K=28$
- 10 exotic cycles for $a=7$ at $K=30$

```bash
python scripts/run_all_verifications.py
```

---

## Key Results

### Phase Diagram

| Type | $a$ | Spectral Gap | Max Eigenvalue | Basin of 1 | Limit Behavior |
|------|-----|--------------|----------------|------------|----------------|
| I | 3 | Yes | 0.750 | 100% | Convergence |
| II | 5 | Marginal | 1.000 | Mixed | Indeterminate |
| III | 7 | No | 3.5 | 0.003% | Divergence |

### Collatz ($a=3$) Critical Transition at K=13

| K | # Cycles | Dominant λ | Drift | Basin Size |
|---|----------|-----------|-------|-----------|
| 12 | 3 | 1.336 | +0.376 | 98.78% |
| 13 | 1 | 0.750 | -0.415 | **100%** |
| 20 | 2 | 0.75 / 1.166 | -0.415 / +0.221 | 99.95% |
| 30 | 1 | 0.750 | -0.415 | 100% |

### Ultra-Champion ($a=7$) at K=30

- **10** exotic cycles, max length 9,747
- **Max eigenvalue:** 3.5
- **Dynamics:** 99.997% in exotic cycles, 0.003% in basin of 1
- **Verification:** Cross-confirmed via three independent implementations

---

## Generated Figures

After running `python src/python/plots.py`:

- `phase_diagram.png` — Three-type classification in parameter space
- `cycle_evolution_a3.png` — Cycle count vs K (Collatz)
- `cycle_evolution_a7.png` — Cycle count vs K (Ultra-Champion)
- `k20_block_structure.png` — K=20 resonance three-block structure
- `basin_evolution.png` — Basin dynamics for $a=3$
- `eigenvalue_spectrum.png` — Spectral comparison ($a=3,5,7$)
- `ipr_localization.png` — Inverse Participation Ratio (point localization)

---

## Citation

```bibtex
@unpublished{nieto2026thermodynamic,
  title={Thermodynamic Confinement in Discrete Dynamical Systems v4.0: 
         Spectral Phase Transition in the Family of Accelerated Collatz Maps},
  author={Nieto, Luciano Benjamín},
  year={2026},
  note={DDSD Part 3, Computationally Verified up to K=30}
}
```

---

## Epistemological Status

This repository reports computational observations, not mathematical theorems (excepting Theorem 2.3). The phase diagram in $a$ is a conjecture supported by computational evidence. See the paper's Section 10 for a rigorous separation of proven vs. open questions.

---

## License

MIT License — See [LICENSE](LICENSE) for full terms.

---

*Per Aspera, Ad Astra.*
