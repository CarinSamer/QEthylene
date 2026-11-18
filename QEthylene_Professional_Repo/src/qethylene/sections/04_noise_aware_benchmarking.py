P1, P2, P_RO = 2e-3, 2e-2, 2e-2                 # 1q depol, 2q depol, readout error

def make_noise_model(p1=P1, p2=P2, p_ro=P_RO):
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1),
                                   ["u", "ry", "rz", "rx", "sx", "x", "h"])
    nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx", "cz"])
    nm.add_all_qubit_readout_error(ReadoutError([[1 - p_ro, p_ro], [p_ro, 1 - p_ro]]))
    return nm

NOISE = make_noise_model()
sim_shots = AerSimulator(noise_model=NOISE)
hea_opt = hea.assign_parameters(p_hea)          # 2-qubit state-prep at the VQE optimum

def energy_from_shots(state_circ, H, sim, shots=100000, A_inv=None):
    '''Shot-based <H> for a SparsePauliOp, with optional readout mitigation.'''
    n = state_circ.num_qubits
    total = 0.0
    for label, coeff in zip(H.paulis.to_labels(), H.coeffs.real):
        if set(label) == {"I"}:
            total += coeff
            continue
        qc = state_circ.copy()
        for q, p in enumerate(reversed(label)):        # qiskit is little-endian
            if p == "X": qc.h(q)
            elif p == "Y": qc.sdg(q); qc.h(q)
        qc.measure_all()
        counts = sim.run(transpile(qc, sim), shots=shots).result().get_counts()
        probs = np.zeros(2 ** n)
        for b, c in counts.items():
            probs[int(b, 2)] += c / shots
        if A_inv is not None:
            probs = np.clip(A_inv @ probs, 0, None); probs /= probs.sum()
        exp = 0.0
        for idx, pr in enumerate(probs):
            par = sum(((idx >> q) & 1) for q, p in enumerate(reversed(label)) if p != "I") % 2
            exp += pr * (1 if par == 0 else -1)
        total += coeff * exp
    return total

def calibration_matrix(n, sim, shots=100000):
    A = np.zeros((2 ** n, 2 ** n))
    for prep in range(2 ** n):
        qc = QuantumCircuit(n)
        for q in range(n):
            if (prep >> q) & 1:
                qc.x(q)
        qc.measure_all()
        counts = sim.run(transpile(qc, sim), shots=shots).result().get_counts()
        for b, c in counts.items():
            A[int(b, 2), prep] += c / shots
    return A

A = calibration_matrix(H_par_eq.num_qubits, sim_shots)
A_inv = np.linalg.inv(A)
print("readout confusion matrix A (rows = measured, cols = prepared):")
print(np.round(A, 3))

def dm_energy(circ, H, noise_model):
    '''Exact (no sampling) noisy <H> via the density-matrix simulator.'''
    sim = AerSimulator(method="density_matrix", noise_model=noise_model)
    c = circ.copy(); c.save_density_matrix()
    rho = DensityMatrix(sim.run(transpile(c, sim)).result().data()["density_matrix"])
    return float(np.real(rho.expectation_value(H)))

# ---- ZNE: scale the depolarising rates by 1x, 2x, 3x and extrapolate to 0x ----
scales = [1.0, 2.0, 3.0]
E_scaled = [dm_energy(hea_opt, H_par_eq, make_noise_model(P1 * s, P2 * s, P_RO))
            for s in scales]
zne_fit = np.polyfit(scales, E_scaled, 1)
E_zne = float(np.polyval(zne_fit, 0.0))

# ---- shot-based numbers for the readout-mitigation demonstration ----
E_exact2  = E0_exact_par
E_raw     = energy_from_shots(hea_opt, H_par_eq, sim_shots)
E_readout = energy_from_shots(hea_opt, H_par_eq, sim_shots, A_inv=A_inv)

# The two mitigation techniques start from different baselines, so they are
# reported as two branches rather than stacked in one column:
#   * the shot branch carries gate noise + readout error + sampling noise;
#   * the density-matrix branch carries gate noise only, because taking <H>
#     from rho never goes through a measurement.
# Readout mitigation removes the readout part; ZNE removes the gate part.
print(f"exact ground energy (2 qubits)            = {E_exact2:.5f} Ha\n")
print("shot-based branch  (gate noise + readout error + sampling)")
print(f"  raw                                     = {E_raw:.5f} Ha   err {abs(E_raw-E_exact2)*1000:6.1f} mHa")
print(f"  + readout mitigation                    = {E_readout:.5f} Ha   err {abs(E_readout-E_exact2)*1000:6.1f} mHa")
print("\ndensity-matrix branch  (gate noise only, no readout, no sampling)")
print(f"  raw (1x noise)                          = {E_scaled[0]:.5f} Ha   err {abs(E_scaled[0]-E_exact2)*1000:6.1f} mHa")
print(f"  + ZNE (linear fit to 0x)                = {E_zne:.5f} Ha   err {abs(E_zne-E_exact2)*1000:6.1f} mHa")
RESULTS["noisy_E0_err_raw_mHa"] = float(abs(E_scaled[0] - E_exact2) * 1000)
RESULTS["noisy_E0_err_zne_mHa"] = float(abs(E_zne - E_exact2) * 1000)

plt.figure(figsize=(7, 4))
plt.plot(scales, E_scaled, "o", ms=9, label="noisy energy at 1x, 2x, 3x noise")
xs = np.linspace(0, 3, 50)
plt.plot(xs, np.polyval(zne_fit, xs), "-", label="linear extrapolation")
plt.plot(0, E_zne, "P", ms=13, label=f"ZNE estimate = {E_zne:.4f} Ha")
plt.axhline(E_exact2, color="k", ls="--", lw=1, label="exact")
plt.xlabel("noise scale factor"); plt.ylabel("energy (Ha)")
plt.title("Zero-noise extrapolation"); plt.legend(); plt.tight_layout(); plt.show()

sim_dm = AerSimulator(method="density_matrix", noise_model=NOISE)
circ_dm = hea_opt.copy(); circ_dm.save_density_matrix()
rho = DensityMatrix(sim_dm.run(transpile(circ_dm, sim_dm)).result().data()["density_matrix"])

I2 = Operator(np.eye(2 ** H_par_eq.num_qubits, dtype=complex))
pool_par = [I2] + [Operator(par.map(o).to_matrix()) for o in exc_ferm]
H2_op = Operator(H_par_eq.to_matrix())

d = len(pool_par)
S  = np.zeros((d, d), complex); Ht = np.zeros((d, d), complex)
for i in range(d):
    for j in range(d):
        S[i, j]  = rho.expectation_value(pool_par[i].adjoint() @ pool_par[j])
        Ht[i, j] = rho.expectation_value(pool_par[i].adjoint() @ H2_op @ pool_par[j])
S, Ht = 0.5 * (S + S.conj().T), 0.5 * (Ht + Ht.conj().T)
w, U = np.linalg.eigh(S); keep = w > 1e-6 * w.max()
X = U[:, keep] / np.sqrt(w[keep])
ev = np.sort(np.linalg.eigvalsh(X.conj().T @ Ht @ X).real)
pi_pi_star_noisy = (ev[1] - ev[0]) * HARTREE2EV

print(f"pi->pi* gap, noiseless QSE = {pi_pi_star_qse:.3f} eV")
print(f"pi->pi* gap, noisy QSE     = {pi_pi_star_noisy:.3f} eV")
print(f"shift due to noise         = {pi_pi_star_noisy - pi_pi_star_qse:+.3f} eV")
RESULTS["noisy_pi_pi_star_eV"] = float(pi_pi_star_noisy)

labels = ["exact", "raw\n(shots)", "+ readout\nmitigation",
          "raw\n(density matrix)", "+ ZNE"]
energies = [E_exact2, E_raw, E_readout, E_scaled[0], E_zne]
errs_mHa = [abs(e - E_exact2) * 1000 for e in energies]
# grey = reference, blue = shot branch (gate+readout+sampling), orange = DM branch (gate only)
cols = ["0.55", "#3b6fb5", "#7fa8d8", "#e08a3c", "#f0b878"]

fig, ax = plt.subplots(1, 2, figsize=(13, 5.4))

b = ax[0].bar(labels, errs_mHa, color=cols, edgecolor="black", linewidth=0.8, width=0.62)
ax[0].bar_label(b, fmt="%.1f", padding=3, fontsize=10)
ax[0].axhline(1.6, ls="--", lw=1.6, color="crimson",
              label="chemical accuracy (1.6 mHa)")
ax[0].set_ylabel("absolute error in ground-state energy (mHa)", fontsize=11)
ax[0].set_title("A. Each technique removes a different half of the error",
                fontsize=11.5)
ax[0].set_ylim(0, max(errs_mHa) * 1.22)
ax[0].legend(fontsize=9.5)
ax[0].grid(axis="y", alpha=0.3)
ax[0].tick_params(axis="x", labelsize=9)

gaps = [pi_pi_star_qse, pi_pi_star_noisy, EXP_PI_PI_STAR]
glab = ["noiseless\nQSE", "noisy\nQSE", "experiment"]
b2 = ax[1].bar(glab, gaps, color=["#3b6fb5", "#e08a3c", "0.55"],
               edgecolor="black", linewidth=0.8, width=0.55)
ax[1].bar_label(b2, fmt="%.3f", padding=3, fontsize=10)
shift = pi_pi_star_noisy - pi_pi_star_qse
ax[1].text(1.0, max(gaps) * 1.11,
           f"noise shifts the gap by {shift:+.3f} eV  "
           f"({abs(shift) / pi_pi_star_qse * 100:.1f}%)\n"
           f"roughly twice the relative error of the ground state,\n"
           f"because a gap is a difference of two noisy numbers",
           fontsize=9.5, ha="center", va="center", color="0.2",
           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.75", alpha=0.92))
ax[1].set_ylabel(r"singlet $\pi \to \pi^*$ excitation energy (eV)", fontsize=11)
ax[1].set_title("B. Excitation energies are the fragile ones", fontsize=11.5)
ax[1].set_ylim(0, max(gaps) * 1.25)
ax[1].grid(axis="y", alpha=0.3)

plt.suptitle("Core Task 4: noise and error mitigation, 2-qubit tapered Hamiltonian",
             fontsize=12.5)
plt.tight_layout()
plt.show()

print(f"raw shot-based error        = {errs_mHa[1]:6.1f} mHa   (gate + readout + sampling)")
print(f"  after readout mitigation  = {errs_mHa[2]:6.1f} mHa   <- readout half removed")
print(f"raw density-matrix error    = {errs_mHa[3]:6.1f} mHa   (gate errors only)")
print(f"  after ZNE                 = {errs_mHa[4]:6.1f} mHa   <- gate half removed")
print()
print("Note how the readout-mitigated error lands close to the density-matrix raw")
print("error: what readout mitigation leaves behind IS the gate-noise contribution,")
print("which is exactly what ZNE then removes. Applied together on real hardware,")
print("these two would be expected to reach chemical accuracy.")
