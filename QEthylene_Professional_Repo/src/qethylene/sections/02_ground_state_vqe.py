def exact_expectation(circuit, hamiltonian, theta):
    '''<psi(theta)|H|psi(theta)> from the exact statevector.'''
    bound = circuit.assign_parameters(theta)
    return float(np.real(Statevector(bound).expectation_value(hamiltonian)))

def run_vqe(hamiltonian, ansatz, x0, method="COBYLA", maxiter=400,
            restarts=1, seed=0):
    '''Minimise the energy over the ansatz parameters.
    Returns (best_params, best_energy, energy_history_of_best_run).'''
    rng = np.random.default_rng(seed)
    starts = [np.asarray(x0, float)]
    for _ in range(restarts - 1):
        starts.append(rng.uniform(-np.pi, np.pi, size=ansatz.num_parameters))
    best = None
    for s in starts:
        hist = []
        def cost(x):
            e = exact_expectation(ansatz, hamiltonian, x)
            hist.append(e)
            return e
        res = minimize(cost, s, method=method, options={"maxiter": maxiter})
        if best is None or res.fun < best[1]:
            best = (res.x, res.fun, hist)
    return best

eig_jw = np.linalg.eigvalsh(H_jw_eq.to_matrix())
E0_exact = eig_jw[0]
print(f"exact ground-state energy (active space) = {E0_exact:.10f} Ha")
print("lowest 6 eigenvalues:", np.round(eig_jw[:6], 6))

hf_jw = HartreeFock(cas_eq.num_spatial_orbitals, cas_eq.num_particles, jw)
ucc = UCCSD(cas_eq.num_spatial_orbitals, cas_eq.num_particles, jw,
            initial_state=hf_jw)
x0_ucc = np.zeros(ucc.num_parameters)               # zeros == the Hartree-Fock state
p_ucc, e_ucc, h_ucc = run_vqe(H_jw_eq, ucc, x0_ucc, method="COBYLA",
                              maxiter=500, restarts=1)
ucc_t = transpile(ucc, basis_gates=["cx", "u"], optimization_level=1)

print("UCCSD ansatz")
print(f"  qubits           = {ucc.num_qubits}")
print(f"  parameters       = {ucc.num_parameters}")
print(f"  transpiled depth = {ucc_t.depth()}   (CX gates = {ucc_t.count_ops().get('cx', 0)})")
print(f"  VQE energy       = {e_ucc:.10f} Ha")
print(f"  error vs exact   = {abs(e_ucc - E0_exact):.2e} Ha  "
      f"({abs(e_ucc - E0_exact) * 1000:.4f} mHa)")
RESULTS["VQE_UCCSD_err_mHa"] = float(abs(e_ucc - E0_exact) * 1000)
RESULTS["UCCSD_params"] = int(ucc.num_parameters)


def hardware_efficient(hf_circuit, reps=2):
    nq = hf_circuit.num_qubits
    theta = ParameterVector("t", nq * (reps + 1))
    qc = hf_circuit.copy()
    k = 0
    for r in range(reps + 1):
        for q in range(nq):
            qc.ry(theta[k], q); k += 1
        if r < reps:
            for q in range(nq - 1):
                qc.cx(q, q + 1)
    return qc

E0_exact_par = np.linalg.eigvalsh(H_par_eq.to_matrix()).min()
hf_par = HartreeFock(cas_eq.num_spatial_orbitals, cas_eq.num_particles, par)
hea = hardware_efficient(hf_par, reps=2)
p_hea, e_hea, h_hea = run_vqe(H_par_eq, hea, np.zeros(hea.num_parameters),
                              method="COBYLA", maxiter=400, restarts=6, seed=1)
hea_t = transpile(hea, basis_gates=["cx", "u"], optimization_level=1)
print("Hardware-efficient ansatz")
print(f"  qubits           = {hea.num_qubits}")
print(f"  parameters       = {hea.num_parameters}")
print(f"  transpiled depth = {hea_t.depth()}   (CX gates = {hea_t.count_ops().get('cx', 0)})")
print(f"  VQE energy       = {e_hea:.10f} Ha")
print(f"  error vs exact   = {abs(e_hea - E0_exact_par):.2e} Ha  "
      f"({abs(e_hea - E0_exact_par) * 1000:.4f} mHa)")
RESULTS["VQE_HEA_err_mHa"] = float(abs(e_hea - E0_exact_par) * 1000)
RESULTS["HEA_params"] = int(hea.num_parameters)

plt.figure(figsize=(7, 4))
plt.plot(np.minimum.accumulate(h_ucc),
         label=f"UCCSD (4 qubits, {ucc.num_parameters} params)")
plt.plot(np.minimum.accumulate(h_hea),
         label=f"hardware-efficient (2 qubits, {hea.num_parameters} params)")
plt.axhline(E0_exact, color="k", ls="--", lw=1, label="exact ground energy")
plt.xlabel("optimiser evaluation"); plt.ylabel("best energy so far (Ha)")
plt.title("VQE convergence -- planar ethylene, CAS(2,2)/STO-3G")
plt.legend(); plt.tight_layout(); plt.show()

ucc_depth, ucc_cx = ucc_t.depth(), ucc_t.count_ops().get("cx", 0)
hea_depth, hea_cx = hea_t.depth(), hea_t.count_ops().get("cx", 0)

names = ["UCCSD\n(4 qubits)", "hardware-efficient\n(2 qubits, tapered)"]
err   = [max(RESULTS.get("VQE_UCCSD_err_mHa", np.nan), 1e-9),
         max(RESULTS.get("VQE_HEA_err_mHa", np.nan), 1e-9)]
cx    = [ucc_cx, hea_cx]
cols  = ["#3b6fb5", "#e08a3c"]

fig, ax = plt.subplots(1, 3, figsize=(13.5, 5.0))

b0 = ax[0].bar(names, err, color=cols, edgecolor="black", linewidth=0.8, width=0.55)
ax[0].set_yscale("log")
ax[0].bar_label(b0, fmt="%.1e", padding=3, fontsize=9.5)
ax[0].axhline(1.6, ls="--", lw=1.6, color="crimson", label="chemical accuracy")
ax[0].set_ylabel("error vs exact (mHa, log scale)", fontsize=10.5)
ax[0].set_title("A. Accuracy: both essentially exact", fontsize=11)
ax[0].legend(fontsize=9)
ax[0].grid(axis="y", alpha=0.3)

b1 = ax[1].bar(names, cx, color=cols, edgecolor="black", linewidth=0.8, width=0.55)
ax[1].bar_label(b1, fmt="%d", padding=3, fontsize=10)
ax[1].set_ylabel("two-qubit (CX) gates", fontsize=10.5)
ax[1].set_title("B. Cost: a 28x difference", fontsize=11)
ax[1].set_ylim(0, max(cx) * 1.25)
ax[1].grid(axis="y", alpha=0.3)

# survival probability of the whole circuit at the Section 4 two-qubit error rate
p2 = 2e-2
surv = [100 * (1 - p2) ** n for n in cx]
b2 = ax[2].bar(names, surv, color=cols, edgecolor="black", linewidth=0.8, width=0.55)
ax[2].bar_label(b2, fmt="%.0f%%", padding=3, fontsize=10)
ax[2].set_ylabel("chance of an error-free run (%)", fontsize=10.5)
ax[2].set_title(f"C. Consequence at {p2:.0%} CX error", fontsize=11)
ax[2].set_ylim(0, 108)
ax[2].grid(axis="y", alpha=0.3)

plt.suptitle("Core Task 2: the two ansatze reach the same answer at very "
             "different prices", fontsize=12.5)
plt.tight_layout()
plt.show()

print(f"UCCSD             : {RESULTS.get('UCCSD_params')} parameters, "
      f"depth {ucc_depth}, {ucc_cx} CX  ->  {surv[0]:.0f}% of runs error-free")
print(f"hardware-efficient: {RESULTS.get('HEA_params')} parameters, "
      f"depth {hea_depth}, {hea_cx} CX  ->  {surv[1]:.0f}% of runs error-free")
print()
print("Read panel C carefully before generalising. The hardware-efficient ansatz")
print("wins here partly because CAS(2,2) after tapering is a TWO-qubit problem,")
print("and any two-qubit circuit can reach any two-qubit state. That guarantee")
print("disappears as systems grow, where UCCSD's chemical structure starts to")
print("matter and shallow generic circuits run into barren plateaus.")
