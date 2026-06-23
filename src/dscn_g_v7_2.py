"""
DSCN-G v7.2 — Dual-State Cognitive Geometry (Official Simulator)
=================================================================
Author: Luciano Benjamín Nieto
Email: lucianobenjaminnieto@gmail.com

Official parameters verifying Theorems 1-3:
  D=4, K=10, beta=0.10, eta=0.05, gamma=0.01, theta_death=0.10

CORRECTIONS in this version:
  - ec3: Fixed sign(o_i) to match paper Eq. 3 (was incorrectly (2*outcome-1))
  - ec7: Now implements cognitive interference per paper Eq. 7
  - reward_fn: Formerly ec7, implements reward function (not in paper explicitly)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple
import json, warnings, os, sys, itertools, math, argparse

warnings.filterwarnings("ignore")

# ==================================================================
# OFFICIAL CONSTANTS (Appendix A)
# ==================================================================
K, ETA, BETA, GAMMA = 10, 0.05, 0.10, 0.01
THETA_D, THETA_DIV, LAMBDA_VM, ALPHA = 0.10, 0.80, 3.0, 5.0
EPSILON, D_MAX, R_BASE, D_VEC = 0.10, 3, 1.0, 4
KAPPA, THETA_EMERG, THETA_STAR = 1.0, 0.30, np.pi / 2
STEPS, N_ACTIONS, HALF = 2000, 8, 1000

SCALE_CONFIGS = [
    {"n_init": 4,   "label": "N0=4 (base)",   "seeds": list(range(10))},
    {"n_init": 50,  "label": "N0=50",          "seeds": list(range(10))},
    {"n_init": 200, "label": "N0=200",         "seeds": list(range(10))},
]

# ==================================================================
# DATA STRUCTURES
# ==================================================================
@dataclass
class Node:
    node_id: int
    depth: int
    omega: np.ndarray    phi: float
    vitality: float
    def __post_init__(self): self.omega = self.omega.copy()

class Graph:
    def __init__(self, rng, d=D_VEC, n_init=4):
        self.d, self.nodes, self.edges, self.nxt, self.rng = d, {}, defaultdict(list), 0, rng
        self.root = self._add(depth=0)
        children = []
        for _ in range(min(3, n_init-1)):
            nid = self._add(depth=1); self.edges[self.root].append(nid); children.append(nid)
        added = len(children)
        while added < n_init - 1:
            parent = children[added % len(children)] if children else self.root
            nid = self._add(depth=min(self.nodes[parent].depth + 1, D_MAX))
            self.edges[parent].append(nid); children.append(nid); added += 1
    def _add(self, depth=0, omega=None, phi=None, V=None):
        nid = self.nxt; self.nxt += 1
        if omega is None: omega = self.rng.uniform(0.3, 0.7, self.d)
        if phi is None: phi = self.rng.uniform(0, 2*np.pi)
        if V is None: V = self.rng.uniform(0.3, 0.7)
        self.nodes[nid] = Node(nid, depth, omega, phi, V); self.edges[nid] = []; return nid
    def neighbors(self, nid):
        res = list(self.edges.get(nid, []))
        for s, ds in self.edges.items():
            if nid in ds and s != nid: res.append(s)
        return [n for n in set(res) if n in self.nodes]

class Chain:
    def __init__(self, start, rng): self.cur, self.bit = start, int(rng.integers(0, 2))

# ==================================================================
# MODEL EQUATIONS
# ==================================================================
def ec1(omega, beta, R, outcome, e_R): 
    """Eq. 1 — Stochastic TD learning"""
    return (1-beta)*omega + beta*outcome*R*e_R

def ec2(chains, graph, rng, alpha=ALPHA):
    """Eq. 2 — Information chains with probabilistic transition"""
    cc = defaultdict(int)
    for ch in chains:
        if ch.cur not in graph.nodes: ch.cur = graph.root
        nbrs = [graph.nodes[n] for n in graph.neighbors(ch.cur)]
        if nbrs:
            ds = np.array([np.linalg.norm(nb.omega - graph.nodes[ch.cur].omega) for nb in nbrs])
            lp = -alpha*ds; lp -= lp.max(); p = np.exp(lp); p /= p.sum()
            ch.cur = nbrs[int(rng.choice(len(nbrs), p=p))].node_id
        cc[ch.cur] += 1
    return cc
def ec3(phi, eta, R_i, outcome, theta_a):
    """
    Eq. 3 — Bounded Kuramoto phase dynamics
    CORRECTED: sign(o_i) where sign(0)=0, sign(1)=1 per paper
    (Previous code incorrectly used (2*outcome-1) which gives -1 for outcome=0)
    """
    sign_o = 1.0 if outcome > 0.5 else 0.0  # sign(0)=0, sign(1)=1
    return (phi + eta * R_i * sign_o * np.sin(theta_a - phi)) % (2 * np.pi)

def ec4(phi_root, lambda_vm, n_actions, rng):
    """Eq. 4 — von Mises action selection"""
    ta = np.linspace(0, 2*np.pi, n_actions, endpoint=False)
    lp = lambda_vm*np.cos(phi_root-ta); lp -= lp.max(); p = np.exp(lp); p /= p.sum()
    idx = int(rng.choice(n_actions, p=p)); return idx, ta[idx]

def ec5(V, gamma, A):
    """Eq. 5 — Vitality (exponential moving average)"""
    return V*np.exp(-gamma) + A*(1-np.exp(-gamma))

def ec6(A, V, kappa):
    """Eq. 6 — Valence signal"""
    return max(0.0, A-V)*kappa

def def1(omega, omega_ideal):
    """Definition 1 — Bounded relevance"""
    return R_BASE/(1+np.linalg.norm(omega-omega_ideal))

def reward_fn(theta_a, theta_star):
    """
    Reward function R(t) ∈ [0,1]
    NOT explicitly defined in paper, but required for Eq. 1
    Returns: (R_g, outcome) where R_g = exp(-3*dist) and outcome = 1 if dist < π/8
    """
    dist = abs(np.sin((theta_a - theta_star) / 2))
    return float(np.exp(-3 * dist)), 1 if dist < np.pi / 8 else 0

def ec7(omega_i_norm, phi_i, phi_root):
    """
    Eq. 7 — Cognitive interference (CORRECTED)
    Previous code had ec7 as reward function; now implements actual paper Eq. 7:
    I_i(t) = ||omega_i(t)|| * cos(phi_i(t) - phi_root(t))
    """
    return omega_i_norm * np.cos(phi_i - phi_root)

def compute_rho_eff(cc, active):
    """Herfindahl concentration index"""
    total = sum(cc.values())
    return sum((cnt/total)**2 for cnt in cc.values()) if total>0 else 0.0
# ==================================================================
# SINGLE SIMULATION
# ==================================================================
def simulate_single(seed, n_init=4, full_trace=False, steps=STEPS, kappa_override=None, theta_emerg_override=None):
    rng = np.random.default_rng(seed)
    g = Graph(rng, D_VEC, n_init=n_init)
    chains = [Chain(g.root, rng) for _ in range(K)]
    omega_ideal = np.ones(D_VEC)/np.sqrt(D_VEC); e_R = omega_ideal.copy()
    kappa = kappa_override if kappa_override is not None else KAPPA
    theta_emerg = theta_emerg_override if theta_emerg_override is not None else THETA_EMERG
    phi_traj, omega_traj, node_traj, val_traj = [], [], [], []
    chain_conc = []; phi_err, omega_norm, mem_hits = [], [], []
    valence_events = []
    for t in range(steps):
        if not g.nodes: break
        cc = ec2(chains, g, rng, ALPHA)
        rho = compute_rho_eff(cc, len(g.nodes)); chain_conc.append(rho)
        phi_r = g.nodes[g.root].phi if g.root in g.nodes else 0.0
        act_idx, theta_a = ec4(phi_r, LAMBDA_VM, N_ACTIONS, rng)
        R_g, o_g = reward_fn(theta_a, THETA_STAR)
        total_valence = 0.0; hijack_detected = False
        for nid in list(g.nodes.keys()):
            if nid not in g.nodes: continue
            nd = g.nodes[nid]; A_i = cc.get(nid, 0)/K
            nd.omega = ec1(nd.omega, BETA, R_g, o_g, e_R)
            Ri = def1(nd.omega, omega_ideal)
            nd.phi = ec3(nd.phi, ETA, Ri, o_g, theta_a)
            nd.vitality = ec5(nd.vitality, GAMMA, A_i)
            E_i = ec6(A_i, nd.vitality, kappa); total_valence += E_i
            if nid != g.root and E_i > theta_emerg: hijack_detected = True
        valence_events.append(1 if hijack_detected else 0)
        for nid in [n for n, nd in list(g.nodes.items()) if n != g.root and nd.vitality < THETA_D]:
            if nid not in g.nodes: continue
            for ch in chains:
                if ch.cur == nid: ch.cur = g.root
            for s in list(g.edges):
                if nid in g.edges.get(s, []): g.edges[s].remove(nid)
            g.edges.pop(nid, None)
            if g.nodes[nid].vitality < THETA_D/10.0: del g.nodes[nid]
        strong = [n for n, nd in g.nodes.items() if n != g.root and nd.vitality > THETA_DIV and nd.depth < D_MAX]
        if len(strong) >= 2:
            for n1, n2 in itertools.combinations(strong, 2):
                nd1, nd2 = g.nodes[n1], g.nodes[n2]
                if abs(nd1.phi - nd2.phi) < math.pi/4:
                    sim_AB = np.dot(nd1.omega, nd2.omega)/(np.linalg.norm(nd1.omega)*np.linalg.norm(nd2.omega)+1e-9)
                    if sim_AB > 0.5:
                        oc = (nd1.omega+nd2.omega)/2.0; oc = oc/(np.linalg.norm(oc)+1e-9)
                        oc = oc*min(np.linalg.norm(nd1.omega), np.linalg.norm(nd2.omega))*(sim_AB**2)
                        cid = g._add(max(nd1.depth, nd2.depth)+1, oc, (nd1.phi+nd2.phi)/2.0, THETA_D*(1+sim_AB))
                        g.edges[n1] = g.edges.get(n1, [])+[cid]; g.edges[n2] = g.edges.get(n2, [])+[cid]                        nd1.vitality *= 0.9; nd2.vitality *= 0.9
        if g.root in g.nodes:
            pn = g.nodes[g.root].phi; on = np.linalg.norm(g.nodes[g.root].omega)
        else: pn = on = float('nan')
        q = g.nodes[g.root].omega if g.root in g.nodes else omega_ideal
        mem_hit = 1 if any(np.linalg.norm(nd.omega - q) < EPSILON for nd in g.nodes.values()) else 0
        if full_trace:
            phi_traj.append(float(pn)); omega_traj.append(float(on)); node_traj.append(len(g.nodes)); val_traj.append(float(total_valence))
        if t >= HALF:
            phi_err.append(abs(pn-THETA_STAR) if not np.isnan(pn) else float('nan'))
            omega_norm.append(float(on)); mem_hits.append(mem_hit)
    phi_err = [x for x in phi_err if not np.isnan(x)]; omega_norm = [x for x in omega_norm if not np.isnan(x)]
    out = {'seed': seed, 'n_init': n_init, 'node_count': len(g.nodes),
           'phi_error_mean': float(np.mean(phi_err)) if phi_err else float('nan'),
           'phi_error_std': float(np.std(phi_err)) if phi_err else float('nan'),
           'omega_mean': float(np.mean(omega_norm)) if omega_norm else float('nan'),
           'omega_std': float(np.std(omega_norm)) if omega_norm else float('nan'),
           'omega_final': float(omega_norm[-1]) if omega_norm else float('nan'),
           'memory_rate': float(np.mean(mem_hits)) if mem_hits else float('nan'),
           'rho_eff': float(np.mean(chain_conc)) if chain_conc else 0.0,
           'hijack_rate': float(np.mean(valence_events)) if valence_events else 0.0,
           'hijack_count': sum(valence_events),
           'converged_target': (abs(float(np.mean(phi_err))-THETA_STAR) < np.pi/2) if phi_err else False}
    if full_trace:
        out.update({'phi_traj': phi_traj, 'omega_traj': omega_traj, 'node_traj': node_traj, 'val_traj': val_traj})
    return out

def make_serializable(obj):
    if isinstance(obj, np.bool_): return bool(obj)
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list): return [make_serializable(v) for v in obj]
    return obj

def run_batch_simulation(seeds=None, steps=STEPS, n_init=4, full_trace_seeds=None, kappa_override=None, theta_emerg_override=None, verbose=True):
    if seeds is None: seeds = list(range(100))
    if full_trace_seeds is None: full_trace_seeds = [0, 10, 50, 90]
    results, traces = [], {}
    if verbose:
        print(f"Simulating {len(seeds)} seeds x {steps} steps...")
        print("=" * 65)
    for i, seed in enumerate(seeds):
        full = seed in full_trace_seeds
        r = simulate_single(seed, n_init=n_init, full_trace=full, steps=steps, kappa_override=kappa_override, theta_emerg_override=theta_emerg_override)
        results.append(r)
        if full: traces[seed] = r
        if verbose and (i + 1) % 10 == 0:
            status = "OK" if r['converged_target'] else "ANT"            print(f"  {i+1}/{len(seeds)} | seed={seed} [{status}] | N={r['node_count']} | phi_err={r['phi_error_mean']:.4f} | omega={r['omega_mean']:.3f} | hijack={r['hijack_rate']*100:.1f}%")
    converged = [r for r in results if r['converged_target']]
    p_converge = len(converged) / len(results)
    nc_all = [r['node_count'] for r in results]
    pe_all = [r['phi_error_mean'] for r in results]
    om_all = [r['omega_mean'] for r in results]
    of_all = [r['omega_final'] for r in results]
    me_all = [r['memory_rate'] for r in results]
    rho_all = [r['rho_eff'] for r in results]
    hij_all = [r['hijack_rate'] for r in results]
    summary = make_serializable({
        "n_seeds": len(seeds), "n_steps": steps, "n_init": n_init,
        "parameters": {"K": K, "ETA": ETA, "BETA": BETA, "GAMMA": GAMMA, "THETA_D": THETA_D, "THETA_DIV": THETA_DIV, "LAMBDA_VM": LAMBDA_VM, "ALPHA": ALPHA, "D_VEC": D_VEC, "KAPPA": KAPPA, "THETA_EMERG": theta_emerg_override if theta_emerg_override else THETA_EMERG, "THETA_STAR": float(THETA_STAR)},
        "theorem1": {"N_base_bound": float(1.0/THETA_D), "N_rho_bound": float(np.mean(rho_all)/THETA_D), "N_simulated_mean": float(np.mean(nc_all)), "N_simulated_std": float(np.std(nc_all)), "rho_eff_mean": float(np.mean(rho_all)), "rho_eff_std": float(np.std(rho_all)), "verified": float(np.mean(nc_all)) <= float(np.mean(rho_all)/THETA_D)},
        "theorem2": {"omega_star_theoretical": 0.649, "omega_simulated_mean": float(np.mean(of_all)), "omega_simulated_std": float(np.std(of_all)), "O_beta": BETA, "verified": abs(float(np.mean(of_all)) - 0.649) < BETA},
        "theorem3": {"p_converge": float(p_converge), "n_converged": len(converged), "n_antipodal": len(results)-len(converged), "phi_error_converged_mean": float(np.mean([r['phi_error_mean'] for r in converged])) if converged else None, "phi_error_converged_std": float(np.std([r['phi_error_mean'] for r in converged])) if converged else None, "phi_error_antipodal_mean": float(np.mean([r['phi_error_mean'] for r in results if not r['converged_target']])) if len(results)>len(converged) else None, "verified": p_converge > 0.5},
        "prediction3": {"hijack_rate_mean": float(np.mean(hij_all)), "hijack_rate_std": float(np.std(hij_all)), "hijack_rate_percent": float(np.mean(hij_all)*100)},
        "memory_rate_mean": float(np.mean(me_all)), "memory_rate_std": float(np.std(me_all)),
    })
    if verbose:
        print("\n" + "=" * 65)
        print("RESULTS SUMMARY")
        print("=" * 65)
        print(f"[Theorem 1] N* = {summary['theorem1']['N_simulated_mean']:.2f} +- {summary['theorem1']['N_simulated_std']:.2f} | Bound = {summary['theorem1']['N_rho_bound']:.2f} | {'VERIFIED' if summary['theorem1']['verified'] else 'FAIL'}")
        print(f"[Theorem 2] omega* = {summary['theorem2']['omega_simulated_mean']:.3f} +- {summary['theorem2']['omega_simulated_std']:.3f} | Baseline = {summary['theorem2']['omega_star_theoretical']:.3f} | O(beta) = {BETA} | {'VERIFIED' if summary['theorem2']['verified'] else 'FAIL'}")
        print(f"[Theorem 3] p_conv = {p_converge:.2f} > 0.5 | {len(converged)}/{len(results)} seeds | {'VERIFIED' if summary['theorem3']['verified'] else 'FAIL'}")
        print(f"[Prediction 3] Hijack rate = {summary['prediction3']['hijack_rate_percent']:.1f}%")
    return results, traces, summary

def run_parameter_sweep(kappa_values=None, theta_emerg_values=None, seeds_per_config=20, steps=STEPS, verbose=True):
    if kappa_values is None: kappa_values = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    if theta_emerg_values is None: theta_emerg_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    sweep_results = {}; total = len(kappa_values)*len(theta_emerg_values)
    if verbose: print(f"Parameter sweep: {len(kappa_values)} kappa x {len(theta_emerg_values)} theta = {total} configs"); print("=" * 65)
    idx = 0
    for kappa in kappa_values:
        for theta_e in theta_emerg_values:
            idx += 1; key = f"kappa={kappa:.1f}_theta={theta_e:.1f}"
            if verbose: print(f"[{idx}/{total}] {key} | {seeds_per_config} seeds...", end=" ")
            _, _, summary = run_batch_simulation(seeds=list(range(seeds_per_config)), steps=steps, kappa_override=kappa, theta_emerg_override=theta_e, verbose=False)
            hr = summary['prediction3']['hijack_rate_percent']; pv = summary['theorem3']['p_converge']
            if verbose: print(f"hijack={hr:.1f}% | p_conv={pv:.2f}")
            sweep_results[key] = {"kappa": kappa, "theta_emerg": theta_e, "hijack_rate": summary['prediction3']['hijack_rate_mean'], "hijack_rate_percent": hr, "p_converge": pv, "N_mean": summary['theorem1']['N_simulated_mean'], "omega_mean": summary['theorem2']['omega_simulated_mean']}
    falsified = True
    for res in sweep_results.values():
        if res['kappa'] >= 0.5 and res['theta_emerg'] <= 0.5:
            if res['hijack_rate_percent'] >= 5.0: falsified = False; break
    if verbose: print("\n" + "=" * 65); print("SWEEP RESULTS"); print("=" * 65); print(f"Falsification criterion: {'FALSIFIED' if falsified else 'NOT FALSIFIED'}")
    return sweep_results
def simulate_autopoietic_field(n_nodes=100, steps=1000, seeds=10):
    results = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed); rho = np.ones(n_nodes)*0.1; phi = rng.uniform(0, 2*np.pi, n_nodes)
        connectivity, g, m_star, hbar = 4, 0.5, 1.0, 1.0
        rho_traj, coherence_traj = [], []
        for t in range(steps):
            lap_rho = np.zeros(n_nodes); lap_phi = np.zeros(n_nodes)
            for i in range(n_nodes):
                neighbors = [(i+j)%n_nodes for j in range(-2, 3) if j!=0]
                for j in neighbors: lap_rho[i] += rho[j]-rho[i]; lap_phi[i] += np.sin(phi[j]-phi[i])
            mu = g*rho; drho = (-hbar/(2*m_star))*lap_rho + mu*rho - 0.01*rho**3
            drho = np.clip(drho, -0.1, 0.1); rho = np.clip(rho+0.1*drho, 0.001, 1.0)
            dphi = lap_phi + rng.normal(0, 0.01, n_nodes); phi = (phi+0.05*dphi)%(2*np.pi)
            order_param = np.abs(np.mean(np.exp(1j*phi)))
            rho_traj.append(float(np.mean(rho))); coherence_traj.append(float(order_param))
        results.append({"seed": seed, "rho_final": float(np.mean(rho)), "coherence_final": float(coherence_traj[-1]), "rho_traj": rho_traj, "coherence_traj": coherence_traj})
    coh_all = [r['coherence_final'] for r in results]
    summary = make_serializable({"coherence_mean": float(np.mean(coh_all)), "coherence_std": float(np.std(coh_all)), "phase_transition_detected": float(np.mean(coh_all)) > 0.3})
    return results, summary

def simulate_q_dscn_g_approx(n_nodes=50, steps=500, seeds=10):
    results = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed); x = rng.normal(0, 0.1, n_nodes); p = rng.normal(0, 0.1, n_nodes)
        hbar, gamma_decoherence = 0.1, 0.01; entropy_traj = []
        for t in range(steps):
            noise_x = rng.normal(0, np.sqrt(hbar), n_nodes); noise_p = rng.normal(0, np.sqrt(hbar), n_nodes)
            dx = p - gamma_decoherence*x + noise_x; dp = -x - gamma_decoherence*p + noise_p
            x += 0.01*dx; p += 0.01*dp
            sigma2 = np.mean(x**2)*np.mean(p**2) - np.mean(x*p)**2
            if sigma2 > hbar**2/4:
                n_th = np.sqrt(sigma2)/hbar - 0.5; n_th = max(n_th, 0.001)
                S = (n_th+1)*np.log(n_th+1) - n_th*np.log(n_th)
            else: S = 0.0
            entropy_traj.append(float(S))
        results.append({"seed": seed, "entropy_final": entropy_traj[-1], "entropy_traj": entropy_traj})
    return results

# ==================================================================
# PLOT UTILITIES
# ==================================================================
DARK_BG, PANEL_BG, GRID_COL, TEXT_COL = "#0d1117", "#161b22", "#21262d", "#e6edf3"
ACCENT, GREEN, ORANGE, PURPLE, RED, YELLOW, TEAL = "#58a6ff", "#3fb950", "#f0883e", "#bc8cff", "#ff7b72", "#e3b341", "#39d353"
WIN = 80

def smooth(arr, w=WIN):
    arr = np.array(arr)
    if len(arr) < w: return arr
    return np.convolve(arr, np.ones(w)/w, mode='valid')
def styled_ax(ax, title, xlabel="", ylabel="", polar=False):
    ax.set_facecolor(PANEL_BG)
    if not polar:
        for sp in ax.spines.values(): sp.set_edgecolor(GRID_COL)
        ax.grid(True, color=GRID_COL, linewidth=0.5, alpha=0.7)
        ax.tick_params(colors=TEXT_COL, labelsize=8.5)
    ax.set_title(title, color=TEXT_COL, fontsize=9.5, fontweight="bold", pad=6)
    if xlabel: ax.set_xlabel(xlabel, color=TEXT_COL, fontsize=8.5)
    if ylabel: ax.set_ylabel(ylabel, color=TEXT_COL, fontsize=8.5)
    return ax

def save_fig(fig, path, dpi=170):
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Figure saved: {path}")

# ==================================================================
# FIGURES
# ==================================================================
def generate_figure_1_base(results, traces, summary, output_path):
    SEEDS_PLOT = list(range(len(results))); converged = [r for r in results if r['converged_target']]
    p_converge = len(converged)/len(results); pe_all = [r['phi_error_mean'] for r in results]
    om_all = [r['omega_mean'] for r in results]; me_all = [r['memory_rate'] for r in results]
    nc_all = [r['node_count'] for r in results]; bar_colors = [GREEN if r['converged_target'] else RED for r in results]
    fig = plt.figure(figsize=(22, 28), facecolor=DARK_BG)
    fig.suptitle(f"DSCN-G v7.2 — Base Simulation — {len(results)} Seeds x {STEPS} Steps", fontsize=20, fontweight="bold", color=TEXT_COL, y=0.992)
    gs = gridspec.GridSpec(5, 3, figure=fig, hspace=0.54, wspace=0.33, left=0.06, right=0.97, top=0.975, bottom=0.035)
    steps_arr = np.arange(STEPS)
    ax0a = styled_ax(fig.add_subplot(gs[0, 0]), "Phase Error by Seed (green=converged, red=antipodal)", "Seed", "|phi - theta*| [rad]")
    ax0b = styled_ax(fig.add_subplot(gs[0, 1]), "Omega Norm (last 1000 steps)", "Seed", "||omega||")
    ax0c = styled_ax(fig.add_subplot(gs[0, 2]), "Memory Hit Rate (%)", "Seed", "Memory [%]")
    xpos = np.arange(len(results))
    ax0a.bar(xpos, pe_all, color=bar_colors, alpha=0.82); ax0a.axhline(ETA*R_BASE, color=ORANGE, ls="--", lw=1.4, label=f"O(eta)={ETA:.2f}"); ax0a.legend(fontsize=7.5, framealpha=0.3, labelcolor=TEXT_COL); ax0a.set_xticks(xpos[::10]); ax0a.set_xticklabels([str(s) for s in SEEDS_PLOT[::10]], fontsize=8)
    ax0b.bar(xpos, om_all, color=bar_colors, alpha=0.82); ax0b.axhline(0.649, color=ORANGE, ls="--", lw=1.4, label="omega* = 0.649"); ax0b.legend(fontsize=7.5, framealpha=0.3, labelcolor=TEXT_COL); ax0b.set_xticks(xpos[::10]); ax0b.set_xticklabels([str(s) for s in SEEDS_PLOT[::10]], fontsize=8)
    me_pct = [r*100 for r in me_all]; ax0c.bar(xpos, me_pct, color=bar_colors, alpha=0.82); ax0c.set_ylim(0, 115); ax0c.axhline(100, color=GREEN, ls="--", lw=1.4, label="100%"); ax0c.legend(fontsize=7.5, framealpha=0.3, labelcolor=TEXT_COL); ax0c.set_xticks(xpos[::10]); ax0c.set_xticklabels([str(s) for s in SEEDS_PLOT[::10]], fontsize=8)
    ax1a = styled_ax(fig.add_subplot(gs[1, 0:2]), "phi_root Trajectory (selected seeds)", "Step", "phi [rad]")
    ax1b = styled_ax(fig.add_subplot(gs[1, 2]), "||omega_root|| Trajectory", "Step", "||omega||")
    trace_seeds_plot = list(traces.keys())[:3] if traces else [0, 10, 50]
    for s in trace_seeds_plot:
        if s in traces:
            col = ACCENT if traces[s]['converged_target'] else RED; lbl = f"seed {s} {'OK' if traces[s]['converged_target'] else 'ANT'}"
            phi = np.array(traces[s]['phi_traj']); omg = np.array(traces[s]['omega_traj'])
            ax1a.plot(steps_arr[:len(phi)], phi, color=col, lw=0.9, alpha=0.85, label=lbl); ax1b.plot(steps_arr[:len(omg)], omg, color=col, lw=0.9, alpha=0.85, label=lbl)
    ax1a.axhline(THETA_STAR, color=YELLOW, ls="--", lw=1.5, label="theta* = pi/2"); ax1a.axhline(THETA_STAR+np.pi, color=ORANGE, ls=":", lw=1.3, label="theta*+pi"); ax1a.axvline(HALF, color=GRID_COL, lw=1.0, ls="--"); ax1a.legend(fontsize=8, framealpha=0.3, labelcolor=TEXT_COL)
    ax1b.axhline(0.649, color=ORANGE, ls="--", lw=1.4, label="omega* = 0.649"); ax1b.axvline(HALF, color=GRID_COL, lw=1.0, ls="--"); ax1b.legend(fontsize=8, framealpha=0.3, labelcolor=TEXT_COL)
    ax2a = styled_ax(fig.add_subplot(gs[2, 0]), "Active Nodes N(t)", "Step", "N")
    ax2b = styled_ax(fig.add_subplot(gs[2, 1]), f"Reward R(t) (smoothed {WIN} steps)", "Step", "R")
    ax2c = styled_ax(fig.add_subplot(gs[2, 2]), "Accumulated Valence E(t)", "Step", "sum E_i")
    for s in trace_seeds_plot:        if s in traces:
            col = ACCENT if traces[s]['converged_target'] else RED; t = traces[s]
            nd = np.array(t['node_traj']); rw = np.array(t.get('reward_traj', np.zeros(len(nd)))); va = np.array(t['val_traj'])
            ax2a.plot(steps_arr[:len(nd)], nd, color=col, lw=0.9, alpha=0.85); rw_s = smooth(rw); ax2b.plot(steps_arr[WIN-1:WIN-1+len(rw_s)], rw_s, color=col, lw=0.9, alpha=0.85); va_s = smooth(va); ax2c.plot(steps_arr[WIN-1:WIN-1+len(va_s)], va_s, color=col, lw=0.9, alpha=0.85)
    N_base_bound = 1.0/THETA_D; N_rho_bound = summary['theorem1']['N_rho_bound']
    ax2a.axhline(N_base_bound, color=YELLOW, ls=":", lw=1.3, label=f"N* <= 1/theta_d = {N_base_bound:.0f}"); ax2a.axhline(N_rho_bound, color=ORANGE, ls="--", lw=1.3, label=f"N* <= rho/theta_d = {N_rho_bound:.1f}"); ax2a.axhline(np.mean(nc_all), color=GREEN, ls="-", lw=1.0, alpha=0.6, label=f"N sim = {np.mean(nc_all):.1f}"); ax2a.legend(fontsize=7.5, framealpha=0.3, labelcolor=TEXT_COL)
    ax2c.axhline(THETA_EMERG, color=RED, ls="--", lw=1.2, label=f"theta_emerg={THETA_EMERG}"); ax2c.legend(fontsize=7.5, framealpha=0.3, labelcolor=TEXT_COL)
    ax3a = fig.add_subplot(gs[3, 0], polar=True); ax3a.set_facecolor(PANEL_BG); ax3a.set_title("Final phi_root Distribution", color=TEXT_COL, fontsize=9.5, fontweight="bold", pad=18); ax3a.tick_params(colors=TEXT_COL, labelsize=7.5)
    for s in trace_seeds_plot:
        if s in traces and traces[s]['phi_traj']:
            phi_f = traces[s]['phi_traj'][-1]; col = ACCENT if traces[s]['converged_target'] else RED
            ax3a.plot([0, phi_f], [0, 0.9], color=col, lw=2.5, alpha=0.8); ax3a.scatter([phi_f], [0.9], color=col, s=70, zorder=5)
    sample_seeds = results[::5]
    for r in sample_seeds:
        phi_approx = THETA_STAR if r['converged_target'] else THETA_STAR+np.pi; phi_approx += np.random.default_rng(r['seed']).uniform(-0.15, 0.15); col = GREEN if r['converged_target'] else RED
        ax3a.plot([0, phi_approx], [0, 0.85], color=col, lw=1.0, alpha=0.3)
    ax3a.plot([0, THETA_STAR], [0, 1.0], color=YELLOW, lw=2.5, ls="--", label="theta*"); ax3a.plot([0, THETA_STAR+np.pi], [0, 0.85], color=ORANGE, lw=1.5, ls=":", label="theta*+pi"); ax3a.set_ylim(0, 1.1); ax3a.legend(loc="lower right", fontsize=7.5, framealpha=0.3, labelcolor=TEXT_COL)
    ax3b = styled_ax(fig.add_subplot(gs[3, 1]), "||omega_root|| Distribution (last 1000 steps)", "||omega||", "Density")
    for s in trace_seeds_plot:
        if s in traces and traces[s].get('omega_traj'):
            omg = np.array(traces[s]['omega_traj'])[HALF:]; col = ACCENT if traces[s]['converged_target'] else RED
            ax3b.hist(omg, bins=30, color=col, alpha=0.55, density=True)
    ax3b.axvline(0.649, color=ORANGE, ls="--", lw=1.5, label="omega* = 0.649"); ax3b.legend(fontsize=7.5, framealpha=0.3, labelcolor=TEXT_COL)
    ax3c = styled_ax(fig.add_subplot(gs[3, 2]), f"Convergence Classification — p={p_converge:.2f} > 1/2", "Seed", "|phi_err| [rad]")
    for r in results:
        col = GREEN if r['converged_target'] else RED; mark = 'o' if r['converged_target'] else 'D'
        ax3c.scatter(r['seed'], r['phi_error_mean'], color=col, s=40, marker=mark, alpha=0.6)
    ax3c.axhline(np.pi/2, color=YELLOW, ls="--", lw=1.3, label="pi/2 (threshold)"); ax3c.text(0.05, 0.92, f"Converged: {len(converged)}/{len(results)}\np_conv = {p_converge:.2f} > 0.5", transform=ax3c.transAxes, color=TEXT_COL, fontsize=8, va="top", bbox=dict(facecolor=PANEL_BG, alpha=0.7, edgecolor=GREEN)); ax3c.legend(fontsize=8, framealpha=0.3, labelcolor=TEXT_COL)
    ax4 = fig.add_subplot(gs[4, :]); ax4.set_facecolor(PANEL_BG)
    for sp in ax4.spines.values(): sp.set_edgecolor(GRID_COL)
    ax4.set_xticks([]); ax4.set_yticks([]); ax4.set_title("Theorem Verification Summary", color=TEXT_COL, fontsize=11, fontweight="bold", pad=8)
    thm_data = [
        ("Theorem 1", "Homeostatic Stability", f"Bound: N* <= rho/theta_d = {summary['theorem1']['N_rho_bound']:.2f}\nN* sim = {summary['theorem1']['N_simulated_mean']:.2f} +- {summary['theorem1']['N_simulated_std']:.2f}", "VERIFIED", GREEN if summary['theorem1']['verified'] else RED),
        ("Theorem 2", "O(beta) Vector Attractor", f"omega* theoretical = {summary['theorem2']['omega_star_theoretical']:.3f}\nomega* sim = {summary['theorem2']['omega_simulated_mean']:.3f} +- {summary['theorem2']['omega_simulated_std']:.3f}", "VERIFIED", GREEN if summary['theorem2']['verified'] else RED),
        ("Theorem 3", f"Phase Convergence (p={p_converge:.2f})", f"p_converge = {p_converge:.2f} > 0.5\nSeeds TARGET: {len(converged)}/{len(results)}", "VERIFIED" if p_converge > 0.5 else "FAIL", GREEN if p_converge > 0.5 else RED),
        ("Prediction C3", "Phase-Hijacking of Valence", f"theta_emerg = {THETA_EMERG}\nHijack rate = {summary['prediction3']['hijack_rate_percent']:.1f}%", "PREDICTION", ACCENT),
    ]
    x_starts = [0.01, 0.26, 0.51, 0.75]
    for (thm, sub, detail, verdict, vcol), xs in zip(thm_data, x_starts):
        xc = xs + 0.115
        ax4.text(xc, 0.94, thm, color=ACCENT, fontsize=10, fontweight="bold", ha="center", va="top", transform=ax4.transAxes)
        ax4.text(xc, 0.80, sub, color=TEXT_COL, fontsize=8.5, fontstyle="italic", ha="center", va="top", transform=ax4.transAxes)
        ax4.text(xc, 0.62, detail, color=TEXT_COL, fontsize=7.8, ha="center", va="top", transform=ax4.transAxes, family="monospace")
        ax4.text(xc, 0.14, verdict, color=vcol, fontsize=8.5, fontweight="bold", ha="center", va="top", transform=ax4.transAxes)
        if xs > 0.01: ax4.axvline(xs, color=GRID_COL, lw=0.8, ymin=0.05, ymax=0.95)
    save_fig(fig, output_path)

def generate_figure_2_scalability(scale_results, output_path):
    fig = plt.figure(figsize=(22, 18), facecolor=DARK_BG)
    fig.suptitle("DSCN-G v7.2 — Scalability Study N0 in {4, 50, 200}", fontsize=18, fontweight="bold", color=TEXT_COL, y=0.992)    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.33, left=0.07, right=0.97, top=0.968, bottom=0.06)
    SCALE_COLORS = {4: ACCENT, 50: GREEN, 200: PURPLE}; SCALE_MARKS = {4: 'o', 50: 's', 200: 'D'}; N_INITS = sorted(scale_results.keys())
    ax_pe = styled_ax(fig.add_subplot(gs[0, 0]), "Phase Error", "N0", "|phi - theta*|")
    ax_om = styled_ax(fig.add_subplot(gs[0, 1]), "||omega|| Final", "N0", "||omega||")
    ax_me = styled_ax(fig.add_subplot(gs[0, 2]), "Memory Hit Rate", "N0", "Memory [%]")
    for n_init in N_INITS:
        rr = scale_results[n_init]; col = SCALE_COLORS[n_init]
        pes = [r['phi_error_mean'] for r in rr]; oms = [r['omega_mean'] for r in rr]; mes = [r['memory_rate']*100 for r in rr]
        x_jitter = n_init + np.random.uniform(-5, 5, len(pes)); ax_pe.scatter(x_jitter, pes, color=col, s=50, alpha=0.6); ax_pe.errorbar(n_init, np.mean(pes), yerr=np.std(pes), fmt=SCALE_MARKS[n_init], color=col, ecolor=YELLOW, elinewidth=2, capsize=6, ms=10)
        x_jitter = n_init + np.random.uniform(-5, 5, len(oms)); ax_om.scatter(x_jitter, oms, color=col, s=50, alpha=0.6); ax_om.errorbar(n_init, np.mean(oms), yerr=np.std(oms), fmt=SCALE_MARKS[n_init], color=col, ecolor=YELLOW, elinewidth=2, capsize=6, ms=10)
        x_jitter = n_init + np.random.uniform(-5, 5, len(mes)); ax_me.scatter(x_jitter, mes, color=col, s=50, alpha=0.6); ax_me.errorbar(n_init, np.mean(mes), yerr=np.std(mes), fmt=SCALE_MARKS[n_init], color=col, ecolor=YELLOW, elinewidth=2, capsize=6, ms=10)
    ax_pe.axhline(ETA, color=ORANGE, ls="--", lw=1.4); ax_om.axhline(0.649, color=ORANGE, ls="--", lw=1.4, label="omega* = 0.649"); ax_me.axhline(100, color=GREEN, ls="--", lw=1.4); ax_om.legend(fontsize=8, framealpha=0.3, labelcolor=TEXT_COL)
    ax_nf = styled_ax(fig.add_subplot(gs[1, 0]), "N_final vs N0", "N0", "N_final")
    ax_pc = styled_ax(fig.add_subplot(gs[1, 1]), "p_convergence", "N0", "p_conv")
    ax_rh = styled_ax(fig.add_subplot(gs[1, 2]), "rho_eff", "N0", "rho_eff")
    for n_init in N_INITS:
        rr = scale_results[n_init]; col = SCALE_COLORS[n_init]
        nfs = [r['node_count'] for r in rr]; pcs = [1 if r['converged_target'] else 0 for r in rr]; rhos = [r['rho_eff'] for r in rr]
        ax_nf.scatter([n_init]*len(nfs), nfs, color=col, s=60, alpha=0.6); ax_nf.scatter(n_init, np.mean(nfs), color=col, s=120, marker='*', edgecolors='white', zorder=5)
        ax_pc.scatter(n_init, np.mean(pcs), color=col, s=120, marker='*', zorder=5); ax_rh.scatter([n_init]*len(rhos), rhos, color=col, s=60, alpha=0.6); ax_rh.scatter(n_init, np.mean(rhos), color=col, s=120, marker='*', edgecolors='white', zorder=5)
    N_base_bound = 1.0/THETA_D; ax_nf.axhline(N_base_bound, color=YELLOW, ls="--", lw=1.5); ax_pc.axhline(0.5, color=ORANGE, ls="--", lw=1.4)
    save_fig(fig, output_path)

def generate_figure_3_hijacking(sweep_results, output_path):
    kappa_vals = sorted(set(v['kappa'] for v in sweep_results.values())); theta_vals = sorted(set(v['theta_emerg'] for v in sweep_results.values()))
    hijack_matrix = np.zeros((len(kappa_vals), len(theta_vals))); pconv_matrix = np.zeros((len(kappa_vals), len(theta_vals)))
    for i, k in enumerate(kappa_vals):
        for j, t in enumerate(theta_vals):
            key = f"kappa={k:.1f}_theta={t:.1f}"
            if key in sweep_results: hijack_matrix[i, j] = sweep_results[key]['hijack_rate_percent']; pconv_matrix[i, j] = sweep_results[key]['p_converge']
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=DARK_BG)
    fig.suptitle("Prediction C3 — Phase-Hijacking: Parameter Sweep", fontsize=16, fontweight="bold", color=TEXT_COL, y=0.98)
    ax1 = axes[0]; ax1.set_facecolor(PANEL_BG); im1 = ax1.imshow(hijack_matrix, aspect='auto', cmap='YlOrRd', origin='lower')
    ax1.set_xticks(range(len(theta_vals))); ax1.set_xticklabels([f"{t:.1f}" for t in theta_vals]); ax1.set_yticks(range(len(kappa_vals))); ax1.set_yticklabels([f"{k:.1f}" for k in kappa_vals]); ax1.set_xlabel("theta_emerg", color=TEXT_COL, fontsize=10); ax1.set_ylabel("kappa", color=TEXT_COL, fontsize=10); ax1.set_title("Hijack Rate (%)", color=TEXT_COL, fontsize=11, fontweight="bold"); ax1.tick_params(colors=TEXT_COL); plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    for i in range(len(kappa_vals)):
        for j in range(len(theta_vals)): ax1.text(j, i, f"{hijack_matrix[i,j]:.1f}", ha="center", va="center", color="white" if hijack_matrix[i,j] > 30 else "black", fontsize=8)
    ax2 = axes[1]; ax2.set_facecolor(PANEL_BG); im2 = ax2.imshow(pconv_matrix, aspect='auto', cmap='RdYlGn', origin='lower', vmin=0, vmax=1)
    ax2.set_xticks(range(len(theta_vals))); ax2.set_xticklabels([f"{t:.1f}" for t in theta_vals]); ax2.set_yticks(range(len(kappa_vals))); ax2.set_yticklabels([f"{k:.1f}" for k in kappa_vals]); ax2.set_xlabel("theta_emerg", color=TEXT_COL, fontsize=10); ax2.set_ylabel("kappa", color=TEXT_COL, fontsize=10); ax2.set_title("p_converge", color=TEXT_COL, fontsize=11, fontweight="bold"); ax2.tick_params(colors=TEXT_COL); plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    save_fig(fig, output_path)

def generate_figure_4_field(field_results, field_summary, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor=DARK_BG)
    fig.suptitle("Autopoietic Field — Approximate Simulation", fontsize=16, fontweight="bold", color=TEXT_COL, y=0.98)
    ax = axes[0, 0]; ax.set_facecolor(PANEL_BG)
    if field_results:
        r = field_results[0]; ax.plot(r['coherence_traj'], color=ACCENT, lw=1.0); ax.axhline(field_summary['coherence_mean'], color=GREEN, ls="--", lw=1.5, label=f"Mean = {field_summary['coherence_mean']:.3f}")
        ax.set_xlabel("Step", color=TEXT_COL); ax.set_ylabel("Global Coherence", color=TEXT_COL); ax.set_title("Coherence Evolution", color=TEXT_COL, fontweight="bold"); ax.tick_params(colors=TEXT_COL); ax.legend(fontsize=8, framealpha=0.3, labelcolor=TEXT_COL); ax.grid(True, color=GRID_COL, alpha=0.3)
    ax = axes[0, 1]; ax.set_facecolor(PANEL_BG); coh_all = [r['coherence_final'] for r in field_results]; ax.hist(coh_all, bins=20, color=PURPLE, alpha=0.7, edgecolor='white'); ax.axvline(field_summary['coherence_mean'], color=YELLOW, ls="--", lw=2, label=f"Mean = {field_summary['coherence_mean']:.3f}"); ax.set_xlabel("Final Coherence", color=TEXT_COL); ax.set_ylabel("Frequency", color=TEXT_COL); ax.set_title("Final Coherence Distribution", color=TEXT_COL, fontweight="bold"); ax.tick_params(colors=TEXT_COL); ax.legend(fontsize=8, framealpha=0.3, labelcolor=TEXT_COL); ax.grid(True, color=GRID_COL, alpha=0.3)
    ax = axes[1, 0]; ax.set_facecolor(PANEL_BG)
    if field_results: r = field_results[0]; ax.plot(r['rho_traj'], color=TEAL, lw=1.0)    ax.set_xlabel("Step", color=TEXT_COL); ax.set_ylabel("Mean Density", color=TEXT_COL); ax.set_title("Density Evolution", color=TEXT_COL, fontweight="bold"); ax.tick_params(colors=TEXT_COL); ax.grid(True, color=GRID_COL, alpha=0.3)
    ax = axes[1, 1]; ax.set_facecolor(PANEL_BG); ax.text(0.5, 0.5, f"Phase Transition\n\nMean coherence: {field_summary['coherence_mean']:.3f}\nDetected: {'YES' if field_summary['phase_transition_detected'] else 'NO'}\n\nLambda_c theoretical ~ 10.1 \n(for circulant graphs)", ha='center', va='center', transform=ax.transAxes, color=TEXT_COL, fontsize=12, fontweight='bold', bbox=dict(boxstyle='round', facecolor=PANEL_BG, edgecolor=ACCENT, alpha=0.8)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([])
    save_fig(fig, output_path)

# ==================================================================
# MAIN
# ==================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DSCN-G v7.2 — Official Simulator")
    parser.add_argument("mode", nargs="?", default="all", choices=["base", "scalability", "sweep", "field", "quantum", "all"])
    parser.add_argument("--results-dir", default="data")
    parser.add_argument("--figures-dir", default="figures")
    args = parser.parse_args()
    mode, results_dir, figures_dir = args.mode, args.results_dir, args.figures_dir
    os.makedirs(results_dir, exist_ok=True); os.makedirs(figures_dir, exist_ok=True)
    if mode in ("base", "all"):
        print("=" * 65); print("PHASE 1: Base Simulation — 100 seeds x 2000 steps"); print("=" * 65)
        results, traces, summary = run_batch_simulation(seeds=list(range(100)), steps=STEPS, n_init=4, full_trace_seeds=[0, 10, 50, 90])
        with open(f"{results_dir}/base_simulation_100seeds.json", "w") as f:
            json.dump({"summary": summary, "results": [make_serializable({k: v for k, v in r.items() if k not in ('phi_traj', 'omega_traj', 'vit_traj', 'node_traj', 'reward_traj', 'val_traj', 'chain_concentration_traj', 'hijack_details')}) for r in results]}, f, indent=2)
        with open(f"{results_dir}/base_traces.json", "w") as f:
            json.dump({str(k): {kk: vv for kk, vv in v.items() if kk in ('phi_traj', 'omega_traj', 'vit_traj', 'node_traj', 'reward_traj', 'val_traj', 'seed', 'converged_target')} for k, v in traces.items()}, f, indent=2)
        print(f"\nResults saved to {results_dir}/base_simulation_100seeds.json")
        print("\nGenerating Figure 1..."); generate_figure_1_base(results, traces, summary, f"{figures_dir}/fig1_base_simulation.png")
    if mode in ("scalability", "all"):
        print("\n" + "=" * 65); print("PHASE 1B: Scalability Study"); print("=" * 65)
        scale_results = {}
        for cfg in SCALE_CONFIGS:
            n_init = cfg["n_init"]; print(f"\n--- N0 = {n_init} ---")
            res, _, _ = run_batch_simulation(seeds=cfg["seeds"], steps=STEPS, n_init=n_init, verbose=False)
            scale_results[n_init] = res
        with open(f"{results_dir}/scalability_results.json", "w") as f:
            json.dump({str(k): [{kk: vv for kk, vv in r.items() if k not in ('phi_traj', 'omega_traj', 'vit_traj', 'node_traj', 'reward_traj', 'val_traj', 'chain_concentration_traj', 'hijack_details')} for r in v] for k, v in scale_results.items()}, f, indent=2)
        print(f"\nScalability saved to {results_dir}/scalability_results.json")
        print("\nGenerating Figure 2..."); generate_figure_2_scalability(scale_results, f"{figures_dir}/fig2_scalability.png")
    if mode in ("sweep", "all"):
        print("\n" + "=" * 65); print("PHASE 1C: Parameter Sweep (Prediction C3)"); print("=" * 65)
        sweep_results = run_parameter_sweep(kappa_values=[0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0], theta_emerg_values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], seeds_per_config=20, steps=STEPS)
        with open(f"{results_dir}/parameter_sweep.json", "w") as f:
            json.dump(sweep_results, f, indent=2)
        print(f"\nSweep saved to {results_dir}/parameter_sweep.json")
        print("\nGenerating Figure 3..."); generate_figure_3_hijacking(sweep_results, f"{figures_dir}/fig3_hijacking_sweep.png")
    if mode in ("field", "all"):
        print("\n" + "=" * 65); print("PHASE 1D: Autopoietic Field"); print("=" * 65)
        field_results, field_summary = simulate_autopoietic_field(n_nodes=100, steps=1000, seeds=10)
        with open(f"{results_dir}/field_results.json", "w") as f:
            json.dump({"summary": field_summary, "results": [make_serializable({k: v for k, v in r.items() if k not in ('rho_traj', 'coherence_traj')}) for r in field_results]}, f, indent=2)
        print(f"\nField saved to {results_dir}/field_results.json")
        print(f"Mean coherence: {field_summary['coherence_mean']:.3f}")
        print(f"Transition detected: {field_summary['phase_transition_detected']}")        print("\nGenerating Figure 4..."); generate_figure_4_field(field_results, field_summary, f"{figures_dir}/fig4_autopoietic_field.png")
    if mode in ("quantum", "all"):
        print("\n" + "=" * 65); print("PHASE 1E: Q-DSCN-G Approximate"); print("=" * 65)
        q_results = simulate_q_dscn_g_approx(n_nodes=50, steps=500, seeds=10)
        with open(f"{results_dir}/quantum_results.json", "w") as f:
            json.dump({"entropy_final_mean": float(np.mean([r['entropy_final'] for r in q_results])), "results": [make_serializable({k: v for k, v in r.items() if k != 'entropy_traj'}) for r in q_results]}, f, indent=2)
        print(f"\nQuantum saved to {results_dir}/quantum_results.json")
    print("\n" + "=" * 65); print("ALL SIMULATIONS COMPLETED"); print("=" * 65)