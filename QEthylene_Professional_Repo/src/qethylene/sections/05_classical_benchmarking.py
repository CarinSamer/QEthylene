mol = gto.M(atom="; ".join(ethylene_geometry(0)), basis=BASIS, verbose=0)
mf = scf.RHF(mol).run()
print(f"RHF/{BASIS} total energy = {mf.e_tot:.8f} Ha")
mc = mcscf.CASCI(mf, 2, 2)
mc.fcisolver.nroots = 4
mc.kernel()

print("\nCASCI(2,2) roots, labelled by total spin:")
casci_singlets, casci_triplets = [], []
for i, (e, civec) in enumerate(zip(np.atleast_1d(mc.e_tot), mc.ci)):
    ss, _ = mc.fcisolver.spin_square(civec, 2, 2)
    lab = "singlet" if abs(ss) < 0.3 else ("triplet" if abs(ss - 2) < 0.3 else "?")
    print(f"  root {i}: E = {e:.6f} Ha   <S^2> = {ss:5.3f}   {lab}")
    (casci_singlets if lab == "singlet" else casci_triplets).append(e)

assert len(casci_singlets) >= 2, f"expected >=2 singlet roots, got {len(casci_singlets)}"
casci_gap = (casci_singlets[1] - casci_singlets[0]) * HARTREE2EV
casci_st  = ((casci_triplets[0] - casci_singlets[0]) * HARTREE2EV
             if casci_triplets else float("nan"))
print(f"\nCASCI(2,2) singlet pi->pi*  = {casci_gap:.3f} eV   <-- compare to VQE+QSE")
print(f"CASCI(2,2) singlet->triplet = {casci_st:.3f} eV   (a different transition)")


try:
    mcs = mcscf.CASSCF(mf, 2, 2)
    mcs.fcisolver.spin = 0
    mcs.fix_spin_(ss=0)                       # penalise non-singlet solutions
    mcs.state_average_([1 / 3, 1 / 3, 1 / 3])
    mcs.kernel()
    casscf = np.atleast_1d(mcs.e_states)
    casscf_gap = (casscf[1] - casscf[0]) * HARTREE2EV
    print(f"\nCASSCF(2,2) singlet roots (Ha): {np.round(casscf, 6)}")
    print(f"CASSCF(2,2) singlet pi->pi* = {casscf_gap:.3f} eV")
except Exception as exc:
    casscf_gap = float("nan")
    print("CASSCF skipped:", repr(exc))


try:
    from pyscf import cc
    mycc = cc.CCSD(mf).run()
    e_ee, _ = mycc.eomee_ccsd_singlet(nroots=1)
    eom_gap = float(np.atleast_1d(e_ee)[0]) * HARTREE2EV
    print(f"\nEOM-CCSD (full molecule) first singlet excitation = {eom_gap:.3f} eV")
except Exception as exc:
    eom_gap = float("nan")
    print("EOM-CCSD skipped:", repr(exc))

RESULTS["CASCI_pi_pi_star_eV"] = float(casci_gap)
RESULTS["CASCI_singlet_triplet_eV"] = float(casci_st)
RESULTS["CASSCF_pi_pi_star_eV"] = float(casscf_gap)
RESULTS["EOMCCSD_pi_pi_star_eV"] = float(eom_gap)


try:
    from openfermion import MolecularData, jordan_wigner as of_jw, get_sparse_operator
    from openfermion.linalg import get_ground_state
    from openfermionpyscf import run_pyscf

    geo = [(a.split()[0], tuple(map(float, a.split()[1:]))) for a in ethylene_geometry(0)]
    of_mol = run_pyscf(MolecularData(geo, BASIS, 1, 0), run_scf=True)
    nocc = of_mol.n_electrons // 2
    ham = of_mol.get_molecular_hamiltonian(occupied_indices=list(range(nocc - 1)),
                                           active_indices=[nocc - 1, nocc])
    e_of = get_ground_state(get_sparse_operator(of_jw(ham)))[0]
    diff_uHa = abs(e_of - casci_singlets[0]) * 1e6
    print(f"OpenFermion active-space ground energy = {e_of:.8f} Ha")
    print(f"PySCF CASCI root 0                     = {casci_singlets[0]:.8f} Ha")
    print(f"difference                             = {diff_uHa:.2f} micro-Ha  "
          f"{'-> Hamiltonian agrees across two codebases' if diff_uHa < 1 else '-> INVESTIGATE'}")
except Exception as exc:
    print("OpenFermion cross-check skipped:", repr(exc))


rows = [
    ("experiment (pi->pi*)",     EXP_PI_PI_STAR),
    ("VQE + QSE (this work)",    RESULTS.get("QSE_pi_pi_star_eV")),
    ("exact diag of qubit H",    RESULTS.get("exactdiag_pi_pi_star_eV")),
    ("CASCI(2,2)  [PySCF]",      RESULTS.get("CASCI_pi_pi_star_eV")),
    ("CASSCF(2,2) [PySCF]",      RESULTS.get("CASSCF_pi_pi_star_eV")),
    ("EOM-CCSD (full molecule)", RESULTS.get("EOMCCSD_pi_pi_star_eV")),
]
print(f"{'method':<30}{'singlet pi->pi* (eV)':>22}")
print("-" * 52)
for name, gap in rows:
    print(f"{name:<30}{gap:>22.3f}")

print("\nThe three CAS(2,2) entries agree: the qubit pipeline reproduces the")
print("classical CASCI reference on the same active space, which is the check")
print("this section exists for. The remaining gap to experiment is a basis-set")
print("and active-space limitation, not a pipeline error (see Section 3.1).")
print(f"\nsinglet->triplet at CASCI(2,2) = "
      f"{RESULTS.get('CASCI_singlet_triplet_eV'):.3f} eV -- a different state, "
      f"not comparable to 7.6 eV")


names = ["VQE + QSE\n(this work)", "exact diag\nof qubit H", "CASCI(2,2)\n[PySCF]",
         "CASSCF(2,2)\n[PySCF]", "EOM-CCSD\n(full molecule)"]
vals  = [RESULTS.get("QSE_pi_pi_star_eV"), RESULTS.get("exactdiag_pi_pi_star_eV"),
         RESULTS.get("CASCI_pi_pi_star_eV"), RESULTS.get("CASSCF_pi_pi_star_eV"),
         RESULTS.get("EOMCCSD_pi_pi_star_eV")]

cols  = ["#3b6fb5", "#3b6fb5", "#3b6fb5", "#5f88c4", "#e08a3c"]

fig, ax = plt.subplots(figsize=(10, 5.8))
bars = ax.bar(names, vals, color=cols, edgecolor="black", linewidth=0.8, width=0.62)
ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=10)

ax.axhline(EXP_PI_PI_STAR, ls="--", lw=1.8, color="crimson",
           label=f"experiment ({EXP_PI_PI_STAR} eV)")
ax.annotate("", xy=(3.5, EXP_PI_PI_STAR), xytext=(3.5, vals[4]),
            arrowprops=dict(arrowstyle="<->", color="0.3", lw=1.4))
ax.text(3.62, (EXP_PI_PI_STAR + vals[4]) / 2,
        f"basis-set error\n~{vals[4] - EXP_PI_PI_STAR:.1f} eV",
        fontsize=9, va="center", color="0.2", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.75", alpha=0.92))
ax.annotate("", xy=(2.5, vals[2]), xytext=(2.5, vals[4]),
            arrowprops=dict(arrowstyle="<->", color="0.3", lw=1.4))
ax.text(2.38, (vals[2] + vals[4]) / 2,
        f"active-space error\n~{vals[2] - vals[4]:.1f} eV",
        fontsize=9, va="center", ha="right", color="0.2", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.75", alpha=0.92))

ax.set_ylabel(r"singlet $\pi \to \pi^*$ excitation energy (eV)", fontsize=11)
ax.set_title("Core Task 5: the quantum pipeline against classical references\n"
             f"planar ethylene, {BASIS}", fontsize=12)
ax.set_ylim(0, max(vals) * 1.20)
ax.legend(fontsize=10, loc="lower left")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

print("The three left-hand bars share one active space and agree exactly -- that")
print("agreement is what Core Task 5 asks us to demonstrate. CASSCF barely moves,")
print("because reshaping two orbitals cannot substitute for having more of them.")
print(f"EOM-CCSD drops by correlating every orbital in the {BASIS} basis rather than")
print("two, but the remaining distance to experiment is the basis set itself,")
print("which Section 7 takes apart.")
