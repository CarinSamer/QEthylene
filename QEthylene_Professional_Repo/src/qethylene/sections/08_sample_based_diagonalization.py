def det_quantum_numbers(idx, n_spatial=2):
    """Electron count and Sz of a computational-basis determinant.
    Qiskit is little-endian: bit p of `idx` is the occupation of spin-orbital p.
    Ordering is 0..n-1 = alpha, n..2n-1 = beta."""
    occ = [(idx >> p) & 1 for p in range(2 * n_spatial)]
    n_a, n_b = sum(occ[:n_spatial]), sum(occ[n_spatial:])
    return n_a + n_b, 0.5 * (n_a - n_b)


def recover_dets(counts, n_elec=2, n_spatial=2):
    """Configuration recovery: keep only bitstrings in the right N, Sz sector.
    Returns (set of determinant indices, fraction of shots that survived)."""
    kept, total = set(), sum(counts.values())
    good = 0
    for bitstr, c in counts.items():
        idx = int(bitstr.replace(" ", ""), 2)
        n, sz = det_quantum_numbers(idx, n_spatial)
        if n == n_elec and abs(sz) < 1e-9:
            kept.add(idx); good += c
    return kept, (good / total if total else 0.0)


def sqd_spin_resolved(dets, H_matrix, S2_matrix):
    """Project H onto the sampled determinants, diagonalise, and label every
    eigenvector by <S^2> in that same subspace. Returns sorted lists
    (singlet_energies, triplet_energies) and the subspace dimension."""
    idx = sorted(dets)
    if not idx:
        return [], [], 0
    H_sub  = H_matrix[np.ix_(idx, idx)]         # orthonormal basis -> plain eig
    S2_sub = S2_matrix[np.ix_(idx, idx)]
    w, v = np.linalg.eigh(H_sub)
    sing, trip = [], []
    for k in range(len(w)):
        s2 = float(np.real(v[:, k].conj() @ S2_sub @ v[:, k]))
        (sing if s2 < 0.3 else trip).append(float(w[k].real))
    return sorted(sing), sorted(trip), len(idx)


# ---- reference values -------------------------------------------------
Hm_jw = H_jw_eq.to_matrix()
E0_ref  = E0_exact
gap_ref = gap_exact_eV           # spin-filtered exact S0->S1 in the working basis, from Section 3.4

# ---- trial states to sample -------------------------------------------
# Ground state alone spans only {|pi pi>, |pi* pi*>}. Extended SQD samples
# excited trial states too. We use the VQE ground state plus the two
# singly-excited reference determinants (one electron pi -> pi* in each spin
# channel), which is what a single-excitation circuit on top of HF produces.
ucc_state = ucc.assign_parameters(p_ucc)
hf_circ   = HartreeFock(cas_eq.num_spatial_orbitals, cas_eq.num_particles, jw)

exc_alpha = hf_circ.copy(); exc_alpha.x(0); exc_alpha.x(1)   # alpha: pi -> pi*
exc_beta  = hf_circ.copy(); exc_beta.x(2);  exc_beta.x(3)    # beta:  pi -> pi*

TRIALS = {"ground (VQE)": ucc_state, "alpha pi->pi*": exc_alpha, "beta pi->pi*": exc_beta}
sv_trials = {k: Statevector(c) for k, c in TRIALS.items()}

# noisy sampling uses the Section 4 noise model on the same circuits
sim_sqd = AerSimulator(noise_model=NOISE)
qc_trials = {}
for k, c in TRIALS.items():
    q = c.copy(); q.measure_all()
    qc_trials[k] = transpile(q, sim_sqd)

SHOTS = [10, 25, 50, 100, 250, 500, 1000, 4000]
rng_sqd = np.random.default_rng(11)

def run_sqd(shots, noisy):
    """Sample every trial state `shots` times, pool the determinants, solve."""
    dets, kept_frac = set(), []
    for k in TRIALS:
        if noisy:
            counts = sim_sqd.run(qc_trials[k], shots=shots,
                                 seed_simulator=int(rng_sqd.integers(1e6))
                                 ).result().get_counts()
        else:
            counts = sv_trials[k].sample_counts(shots)
        d, f = recover_dets(counts)
        dets |= d; kept_frac.append(f)
    sing, trip, dim = sqd_spin_resolved(dets, Hm_jw, S2m)
    return dict(dim=dim, kept=float(np.mean(kept_frac)),
                e0=sing[0] if sing else np.nan,
                gap=(sing[1] - sing[0]) * HARTREE2EV if len(sing) > 1 else np.nan,
                trip=(trip[0] - sing[0]) * HARTREE2EV if (trip and sing) else np.nan)

rows = []
for sh in SHOTS:
    rows.append((sh, run_sqd(sh, noisy=False), run_sqd(sh, noisy=True)))

print("SQD with extended sampling and spin labelling")
print(f"{'shots':>6} | {'dim':>4}{'E0 err mHa':>12}{'S0->S1 eV':>11}{'S0->T1 eV':>11}"
      f" | {'dim':>4}{'kept':>6}{'E0 err mHa':>12}{'S0->S1 eV':>11}{'S0->T1 eV':>11}")
print(f"{'':>6} | {'---------- noiseless ----------':^38} | "
      f"{'---------- with Section 4 noise ----------':^44}")
print("-" * 98)
for sh, i, n in rows:
    print(f"{sh:>6} | {i['dim']:>4}{abs(i['e0'] - E0_ref) * 1000:>12.3f}"
          f"{i['gap']:>11.3f}{i['trip']:>11.3f}"
          f" | {n['dim']:>4}{n['kept']:>6.0%}{abs(n['e0'] - E0_ref) * 1000:>12.3f}"
          f"{n['gap']:>11.3f}{n['trip']:>11.3f}")

print(f"\nexact:  E0 = {E0_ref:.8f} Ha,  S0->S1 = {gap_ref:.3f} eV")
print(f"QSE (Section 3):    S0->S1 = {pi_pi_star_qse:.3f} eV")
final = rows[-1][1]
print(f"SQD (this section): S0->S1 = {final['gap']:.3f} eV   "
      f"|SQD - QSE| = {abs(final['gap'] - pi_pi_star_qse) * 1000:.2f} meV")
print()
print("The N=2, Sz=0 sector holds 4 determinants and the pooled samples reach all")
print("of them within a few dozen shots, so SQD saturates quickly here. Once the")
print("subspace is complete the projected Hamiltonian is exact, which is why the")
print("noisy column reproduces the noiseless one: noise costs SQD discarded shots")
print("(the 'kept' column), not accuracy. The S0->T1 column is reported so the")
print("triplet is visibly separated from the singlet we are after.")

RESULTS["SQD_E0_err_mHa"]   = float(abs(final["e0"] - E0_ref) * 1000)
RESULTS["SQD_gap_eV"]       = float(final["gap"])
RESULTS["SQD_gap_noisy_eV"] = float(rows[-1][2]["gap"])

# ---------------------------------------------------------------- figure
sh = [r[0] for r in rows]
fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))

ax[0].semilogx(sh, [r[1]["dim"] for r in rows], "o-", lw=2, color="#3b6fb5",
               label="noiseless")
ax[0].semilogx(sh, [r[2]["dim"] for r in rows], "s--", lw=2, color="#e08a3c",
               label="with noise")
ax[0].axhline(4, ls=":", color="0.4", label="full N=2, Sz=0 sector (4 dets)")
ax[0].set_xlabel("shots per trial state"); ax[0].set_ylabel("determinants in the subspace")
ax[0].set_title("A. SQD builds its subspace from samples", fontsize=11.5)
ax[0].set_ylim(0, 4.6); ax[0].legend(fontsize=9.5); ax[0].grid(alpha=0.3)

g_i = [r[1]["gap"] for r in rows]
g_n = [r[2]["gap"] for r in rows]
t_i = [r[1]["trip"] for r in rows]
ax[1].semilogx(sh, g_i, "o-", lw=2, color="#3b6fb5", label="SQD singlet, noiseless")
ax[1].semilogx(sh, g_n, "s--", lw=2, color="#e08a3c", label="SQD singlet, with noise")
ax[1].semilogx(sh, t_i, "^:", lw=1.6, color="0.45", label="SQD triplet (labelled and set aside)")
ax[1].axhline(gap_ref, ls="-", lw=1.4, color="0.3", label=f"exact S0->S1 ({gap_ref:.3f} eV)")
ax[1].axhline(pi_pi_star_qse, ls="-.", lw=1.4, color="#2e9e5b",
              label=f"QSE ({pi_pi_star_qse:.3f} eV)")
ax[1].set_xlabel("shots per trial state"); ax[1].set_ylabel("excitation energy (eV)")
ax[1].set_title("B. SQD finds the singlet, and knows which state is the triplet",
                fontsize=11.5)
ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3, which="both")

plt.suptitle("Sample-based Quantum Diagonalization on the same 4-qubit Hamiltonian",
             fontsize=12.5)
plt.tight_layout()
plt.show()


# --- resource estimation: QSE vs SQD -----------------------------------
# Both prepare the same state, so the circuit cost is identical. The methods
# differ entirely in what they ask for afterwards.
ucc_t2 = transpile(ucc_state, basis_gates=["cx", "u"], optimization_level=1)
n_qubits = H_jw_eq.num_qubits
depth, ncx = ucc_t2.depth(), ucc_t2.count_ops().get("cx", 0)

# --- QSE: every Pauli string appearing in E_mu^dag (H or I) E_nu ------
pool_sp = [SparsePauliOp("I" * n_qubits)] + [jw.map(o) for o in exc_ferm]
qse_paulis = set()
for A in pool_sp:
    for B in pool_sp:
        for prod in (A.adjoint() @ H_jw_eq @ B, A.adjoint() @ B):
            for lbl, c in zip(prod.simplify().paulis.to_labels(),
                              prod.simplify().coeffs):
                if abs(c) > 1e-12 and set(lbl) != {"I"}:
                    qse_paulis.add(lbl)

# Paulis that commute qubit-wise can share one measurement basis: on every
# qubit their letters must either match or one of them must be the identity.
def qwc(a, b):
    return all(x == y or x == "I" or y == "I" for x, y in zip(a, b))

groups = []
for lbl in sorted(qse_paulis):
    for g in groups:
        if all(qwc(lbl, member) for member in g):
            g.append(lbl)
            break
    else:
        groups.append([lbl])
qse_bases = len(groups)

SHOTS_PER_BASIS = 10_000
qse_shots = qse_bases * SHOTS_PER_BASIS
sqd_bases, sqd_shots = 1, SHOTS_PER_BASIS

print(f"{'':<34}{'QSE':>14}{'SQD':>14}")
print("-" * 62)
print(f"{'qubits':<34}{n_qubits:>14}{n_qubits:>14}")
print(f"{'state-prep circuit depth':<34}{depth:>14}{depth:>14}")
print(f"{'two-qubit (CX) gates':<34}{ncx:>14}{ncx:>14}")
print(f"{'distinct Pauli strings to measure':<34}{len(qse_paulis):>14}{0:>14}")
print(f"{'distinct measurement bases':<34}{qse_bases:>14}{sqd_bases:>14}")
print(f"{f'shots at {SHOTS_PER_BASIS:,}/basis':<34}{qse_shots:>14,}{sqd_shots:>14,}")
print(f"{'classical post-processing':<34}{'3x3 gen. eig':>14}{'<=4x4 eig':>14}")
print("-" * 62)
print(f"{'measurement cost ratio':<34}{qse_bases / sqd_bases:>13.0f}x{1:>14}")
print()
print("Both methods run the identical state-preparation circuit, so gate cost is")
print("the same. The difference is entirely in readout: QSE needs a separate")
print("measurement basis for each Pauli group in every matrix element, while SQD")
print("reads the computational basis once. On hardware, where a basis change")
print("means recompiling and requeueing a job, that ratio is the whole story.")

RESULTS["QSE_measurement_bases"] = int(qse_bases)
RESULTS["SQD_measurement_bases"] = int(sqd_bases)

# ---------------------------------------------------------------- figure
fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.0))

m = ["QSE", "SQD"]
b1 = ax[0].bar(m, [qse_bases, sqd_bases], 0.5, color=["#c0504d", "#2e9e5b"],
               edgecolor="black", linewidth=0.9)
ax[0].bar_label(b1, fmt="%d", padding=3, fontsize=11)
ax[0].set_ylabel("distinct measurement bases", fontsize=11)
ax[0].set_title(f"A. Readout cost: {qse_bases / sqd_bases:.0f}x fewer bases for SQD",
                fontsize=11.5)
ax[0].grid(axis="y", alpha=0.3)

gaps = [pi_pi_star_qse, RESULTS["SQD_gap_eV"], gap_ref]
b2 = ax[1].bar(["QSE", "SQD", "exact"], gaps, 0.5,
               color=["#3b6fb5", "#2e9e5b", "0.55"],
               edgecolor="black", linewidth=0.9)
ax[1].bar_label(b2, fmt="%.3f", padding=3, fontsize=11)
ax[1].set_ylabel(r"singlet $\pi\to\pi^*$ (eV)", fontsize=11)
ax[1].set_title("B. Both routes reach the same eigenvalue", fontsize=11.5)
ax[1].set_ylim(0, max(gaps) * 1.2)
ax[1].grid(axis="y", alpha=0.3)

plt.suptitle("QSE vs SQD: same answer, very different measurement cost",
             fontsize=12.5)
plt.tight_layout()
plt.show()
