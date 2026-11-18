def butadiene_geometry():
    return [                                   # planar s-trans 1,3-butadiene (Angstrom)
        "C  -1.833  -0.219  0.000", "C  -0.611   0.325  0.000",
        "C   0.611  -0.325  0.000", "C   1.833   0.219  0.000",
        "H  -2.716   0.406  0.000", "H  -1.944  -1.298  0.000",
        "H  -0.520   1.406  0.000", "H   0.520  -1.406  0.000",
        "H   2.716  -0.406  0.000", "H   1.944   1.298  0.000",
    ]

drv = PySCFDriver(atom=butadiene_geometry(), basis="sto3g", unit=DistanceUnit.ANGSTROM)
bd_full = drv.run()
nocc = bd_full.num_particles[0]
bd_cas = ActiveSpaceTransformer(
    4, 4, active_orbitals=[nocc - 2, nocc - 1, nocc, nocc + 1]).transform(bd_full)
H_bd = jw.map(bd_cas.hamiltonian.second_q_op())


Nm8, Szm8, S2m8 = spin_ops(4, jw)
sing_bd, trip_bd = spin_filtered(H_bd.to_matrix(), Nm8, Szm8, S2m8, n_elec=4)
gap_bd_singlet = (sing_bd[1] - sing_bd[0]) * HARTREE2EV
gap_bd_triplet = (trip_bd[0] - sing_bd[0]) * HARTREE2EV

print(f"butadiene CAS(4,4)/STO-3G -> {H_bd.num_qubits} qubits")
print(f"  lowest singlet excitation = {gap_bd_singlet:.2f} eV   (experiment ~5.9 eV, 1Bu)")
print(f"  lowest triplet excitation = {gap_bd_triplet:.2f} eV   (experiment ~3.2 eV, 3Bu)")

hf_bd = HartreeFock(4, bd_cas.num_particles, jw)
ucc_bd = UCCSD(4, bd_cas.num_particles, jw, initial_state=hf_bd)
_, e_bd, _ = run_vqe(H_bd, ucc_bd, np.zeros(ucc_bd.num_parameters),
                     method="COBYLA", maxiter=300)
print(f"  VQE/UCCSD ground energy   = {e_bd:.6f} Ha  "
      f"(exact {sing_bd[0]:.6f} Ha, error {abs(e_bd - sing_bd[0]) * 1000:.2f} mHa)")
RESULTS["butadiene_singlet_eV"] = float(gap_bd_singlet)
RESULTS["butadiene_triplet_eV"] = float(gap_bd_triplet)


KB_EV, HBAR_EV_S, E_CHG = 8.617333e-5, 6.582119e-16, 1.602177e-19

def marcus_rate(H_DA, dG, lam, T=300.0):
    pref = (2 * np.pi / HBAR_EV_S) * H_DA ** 2
    fc = 1.0 / np.sqrt(4 * np.pi * lam * KB_EV * T)
    return pref * fc * np.exp(-((dG + lam) ** 2) / (4 * lam * KB_EV * T))

H_DA, lam, T = 0.03, 0.50, 300.0            # eV, eV, K   (H_DA & dG would come from QSE)
dG_scan = np.linspace(-1.5, 0.2, 200)
k_scan = np.array([marcus_rate(H_DA, dG, lam, T) for dG in dG_scan])

dG0 = -0.30
k0 = marcus_rate(H_DA, dG0, lam, T)
k_loss, flux = 1e9, 1e8
yield_ = k0 / (k0 + k_loss)
j = E_CHG * flux * yield_
print(f"at dG = {dG0} eV :  k_ET = {k0:.3e} 1/s   (tau = {1 / k0 * 1e12:.1f} ps)")
print(f"transfer yield = {yield_:.4f}   ->   photocurrent j = {j:.3e} A per site")

plt.figure(figsize=(7, 4))
plt.semilogy(-dG_scan, k_scan)
plt.axvline(-dG0, color="grey", ls="--")
plt.xlabel(r"driving force  $-\Delta G$  (eV)"); plt.ylabel(r"$k_{ET}$  (s$^{-1}$)")
plt.title("Marcus electron-transfer rate (illustrative)")
plt.tight_layout(); plt.show()
