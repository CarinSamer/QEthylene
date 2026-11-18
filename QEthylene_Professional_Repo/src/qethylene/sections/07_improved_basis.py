def analyse_orbitals(mol, mf):
    """Per-MO: pi character (out-of-plane population) and spatial extent <r^2>."""
    S   = mol.intor("int1e_ovlp")
    R2  = mol.intor("int1e_r2")
    oop = [i for i, l in enumerate(mol.ao_labels())
           if ("pz" in l) or ("dxz" in l) or ("dyz" in l)]
    nocc = mol.nelectron // 2
    info = []
    for i in range(mf.mo_coeff.shape[1]):
        c = mf.mo_coeff[:, i]
        gross = c * (S @ c)
        info.append(dict(idx=i, occ=(i < nocc), e=float(mf.mo_energy[i]),
                         pi=float(gross[oop].sum() / gross.sum()),
                         r2=float(c @ R2 @ c)))
    return info, nocc


def pick_by_character(info, nocc, thresh=0.80):
    """pi  = highest-energy occupied orbital of pi symmetry.
       pi* = lowest-energy virtual of pi symmetry. The symmetry test excludes a
             3s Rydberg orbital (wrong symmetry for pi*); we do NOT exclude
             diffuse pi orbitals, because the V state genuinely expands in a
             good basis and that expansion is what lowers the energy."""
    occ_pi = [d for d in info if d["occ"]     and d["pi"] > thresh]
    vir_pi = [d for d in info if not d["occ"] and d["pi"] > thresh]
    if not occ_pi or not vir_pi:
        return None, None
    pi = max(occ_pi, key=lambda d: d["e"])
    pistar = min(vir_pi, key=lambda d: d["e"])
    if pistar["e"] > 1.0:
        print(f"    !! pi* candidate at {pistar['e']:.2f} Ha is suspiciously high; "
              f"check the orbital list")
    return pi, pistar


def casci_gap(mf, mol, act):
    """Spin-resolved singlet pi->pi* gap (eV) for a chosen pair of orbitals."""
    mc = mcscf.CASCI(mf, 2, 2)
    mc.fcisolver.nroots = 4
    mo = mc.sort_mo(act, base=0)
    mc.kernel(mo)
    sing, trip = [], []
    for e, civec in zip(np.atleast_1d(mc.e_tot), mc.ci):
        ss, _ = mc.fcisolver.spin_square(civec, 2, 2)
        (sing if abs(ss) < 0.3 else trip).append(e)
    sing, trip = sorted(sing), sorted(trip)
    return ((sing[1] - sing[0]) * HARTREE2EV,
            (trip[0] - sing[0]) * HARTREE2EV if trip else np.nan)


BASES = ["sto-3g", "6-31g", "6-31g*", "6-31+g*", "aug-cc-pvdz"]
scan = []

print("Selecting the active orbitals two ways, in every basis.")
print("'naive' = HOMO and LUMO by index.  'character' = lowest pi-symmetry virtual.\n")
print(f"{'basis':<13}{'AOs':>5}{'qb':>4}  {'naive':>18}   {'character-selected':>26}")
print(f"{'':<13}{'':>5}{'':>4}  {'idx':>7}{'gap eV':>11}   {'idx':>9}{'pi':>6}{'r2*/r2':>8}{'gap eV':>9}")
print("-" * 82)

for b in BASES:
    try:
        mol = gto.M(atom="; ".join(ethylene_geometry(0)), basis=b, verbose=0)
        mf  = scf.RHF(mol).run()
        info, nocc = analyse_orbitals(mol, mf)

        g_naive, _ = casci_gap(mf, mol, [nocc - 1, nocc])           # HOMO / LUMO
        pi, pis = pick_by_character(info, nocc)
        if pi is None:
            print(f"{b:<13}{mol.nao_nr():>5}{4:>4}   no pi-symmetry pair found")
            continue
        g_char, g_trip = casci_gap(mf, mol, [pi["idx"], pis["idx"]])

        same = "" if pis["idx"] != nocc else "  (same as LUMO)"
        print(f"{b:<13}{mol.nao_nr():>5}{4:>4}  "
              f"{str(nocc-1)+'/'+str(nocc):>7}{g_naive:>11.3f}   "
              f"{str(pi['idx'])+'/'+str(pis['idx']):>9}{pis['pi']:>6.2f}"
              f"{pis['r2'] / pi['r2']:>8.2f}{g_char:>9.3f}{same}")
        scan.append((b, mol.nao_nr(), g_naive, g_char, g_trip, pis['r2'] / pi['r2']))
    except Exception as exc:
        print(f"{b:<13}  failed: {type(exc).__name__}: {exc}")

print()
print(f"experiment (band maximum)        = {EXP_PI_PI_STAR} eV")
print("best theoretical vertical values ~ 7.8-8.0 eV")
print("every row above uses CAS(2,2) = 4 qubits. The basis costs nothing in qubits.")

# --------------------------------------------------------------- figure
if scan:
    labs  = [s[0] for s in scan]
    naive = [s[2] for s in scan]
    char  = [s[3] for s in scan]
    ratio = [s[5] for s in scan]
    x = np.arange(len(labs))

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.8))


    ax[0].plot(x, naive, "o--", ms=10, lw=1.8, color="#c0504d", mfc="none", mew=2,
               label="naive: HOMO / LUMO by index")
    ax[0].plot(x, char,  "s-",  ms=8,  lw=2.2, color="#3b6fb5",
               label=r"lowest $\pi$-symmetry virtual")
    ax[0].axhspan(7.6, 8.0, color="green", alpha=0.13,
                  label="experiment / best theory (7.6-8.0 eV)")
    ax[0].axhline(EXP_PI_PI_STAR, ls="--", lw=1.4, color="green")
    for xi, v in zip(x, char):
        ax[0].annotate(f"{v:.2f}", (xi, v), textcoords="offset points",
                       xytext=(0, 11), ha="center", fontsize=9.5, color="#26456e")
    if abs(naive[-1] - char[-1]) > 0.2:
        ax[0].annotate(f"{naive[-1]:.2f}: HOMO+1 is a Rydberg\norbital here, not $\\pi^*$",
                       (x[-1], naive[-1]), textcoords="data", xytext=(x[-1] - 0.55, 12.4),
                       ha="center", fontsize=8.5, color="#8c3936",
                       bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#c0504d", lw=0.8),
                       arrowprops=dict(arrowstyle="-", color="#8c3936", lw=0.8,
                                       connectionstyle="arc3,rad=-0.25"))
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([f"{l}\n({s[1]} AOs)" for l, s in zip(labs, scan)], fontsize=9.5)
    ax[0].set_ylabel(r"singlet $\pi \to \pi^*$ excitation energy (eV)", fontsize=11)
    ax[0].set_title("A. Bigger basis, same 4 qubits", fontsize=11.5)
    ax[0].legend(fontsize=9.5); ax[0].grid(alpha=0.3)


    ax[1].scatter(ratio, char, s=140, color="#3b6fb5", edgecolor="black", zorder=3)
    ax[1].plot(ratio, char, "-", lw=1.4, color="#3b6fb5", alpha=0.6)
    for k, (r, v, l) in enumerate(zip(ratio, char, labs)):
        ax[1].annotate(l, (r, v), textcoords="offset points",
                       xytext=(9, 6 if k % 2 == 0 else -14),
                       fontsize=9.5, color="#26456e")
    ax[1].axhspan(7.6, 8.0, color="green", alpha=0.13)
    ax[1].axhline(EXP_PI_PI_STAR, ls="--", lw=1.4, color="green",
                  label=f"experiment ({EXP_PI_PI_STAR} eV)")
    ax[1].axvline(1.0, ls=":", lw=1.2, color="0.5", label=r"$\pi^*$ same size as $\pi$")
    ax[1].set_xlabel(r"$\langle r^2\rangle_{\pi^*}\;/\;\langle r^2\rangle_{\pi}$"
                     "   (how much the excited orbital has expanded)", fontsize=10.5)
    ax[1].set_ylabel(r"singlet $\pi \to \pi^*$ excitation energy (eV)", fontsize=11)
    ax[1].set_title("B. The mechanism: room to expand lowers the ionic state",
                    fontsize=11.5)
    ax[1].legend(fontsize=9.5, loc="upper right"); ax[1].grid(alpha=0.3)

    plt.suptitle("Closing the gap to experiment without spending qubits", fontsize=12.5)
    plt.tight_layout()
    plt.show()

    print(f"\nSTO-3G -> best basis: {char[0]:.2f} eV -> {char[-1]:.2f} eV "
          f"(moved {char[0] - char[-1]:.2f} eV toward experiment)")
    print(f"pi* extent grew from {ratio[0]:.2f}x to {ratio[-1]:.2f}x the size of pi.")
    print("Panel B is the mechanism: as diffuse functions arrive, pi* swells and the")
    print("ionic V state finally has room. The energy then plateaus at ~8.6 eV: the")
    print("basis has done its job, and the remaining ~0.7 eV to the 7.8-8.0 eV best")
    print("estimates is dynamic correlation, which needs more active orbitals (qubits).")
    if abs(naive[-1] - char[-1]) > 0.2:
        print(f"\nNote the naive rule reads {naive[-1]:.2f} eV in {labs[-1]}: HOMO+1 there is")
        print("an s-type Rydberg orbital, not pi*. That number is a different transition")
        print("that happens to sit near 7.6 eV -- the symmetry check is what catches it.")


BEST = "aug-cc-pvdz"      # basis set for the improved model

mol_b = gto.M(atom="; ".join(ethylene_geometry(0)), basis=BEST, verbose=0)
mf_b  = scf.RHF(mol_b).run()
info_b, nocc_b = analyse_orbitals(mol_b, mf_b)
pi_b, pis_b = pick_by_character(info_b, nocc_b)
act_b = [pi_b["idx"], pis_b["idx"]]
gap_classical, _ = casci_gap(mf_b, mol_b, act_b)

print(f"basis {BEST}: {mol_b.nao_nr()} AOs, active orbitals {act_b} "
      f"(pi char {pi_b['pi']:.2f} / {pis_b['pi']:.2f})")
print(f"classical CASCI(2,2) singlet pi->pi* = {gap_classical:.3f} eV\n")

drv_b  = PySCFDriver(atom=ethylene_geometry(0), basis=BEST,
                     unit=DistanceUnit.ANGSTROM)
full_b = drv_b.run()
cas_b  = ActiveSpaceTransformer(2, 2, active_orbitals=act_b).transform(full_b)
H_b    = jw.map(cas_b.hamiltonian.second_q_op())
Hm_b   = H_b.to_matrix()
print(f"qubits required = {H_b.num_qubits}   (unchanged: the basis costs no qubits)")

sing_b, _ = spin_filtered(Hm_b, Nm, Szm, S2m, n_elec=2)

hf_b  = HartreeFock(2, cas_b.num_particles, jw)
ucc_b = UCCSD(2, cas_b.num_particles, jw, initial_state=hf_b)
p_b, e_b, _ = run_vqe(H_b, ucc_b, np.zeros(ucc_b.num_parameters),
                      method="COBYLA", maxiter=600, restarts=3, seed=3)
psi_b   = Statevector(ucc_b.assign_parameters(p_b)).data
qse_b   = run_qse(Hm_b, psi_b, pool_ops)
gap_qse_b = (qse_b[1] - qse_b[0]) * HARTREE2EV

print(f"\nVQE ground energy   = {e_b:.8f} Ha   "
      f"(exact {sing_b[0]:.8f}, error {abs(e_b - sing_b[0]) * 1000:.2e} mHa)")
print(f"QSE pi->pi*         = {gap_qse_b:.3f} eV")
print(f"classical CASCI     = {gap_classical:.3f} eV")
disagree = abs(gap_qse_b - gap_classical) * 1000
print(f"agreement           = {disagree:.2f} meV")
if disagree > 1.0:
    print()
    print("  Note: these two routes should agree to well under 1 meV. A larger")
    print("  difference means the orbital indices did not carry across between")
    print("  the standalone PySCF run and the driver's own SCF, which happens")
    print("  when two virtual orbitals are near-degenerate. The classical value")
    print("  is the reliable one in that case.")
else:
    print("  agreement confirms the active orbitals carried across correctly")

# ------------------------------------------------------------- figure
# STO-3G and 6-31G values were computed earlier; after the Section 3.4 switch
# RESULTS["QSE_pi_pi_star_eV"] holds the 6-31G number, the STO-3G one was saved
# separately. Labelling both correctly matters.
v_sto = RESULTS.get("QSE_pi_pi_star_sto3g_eV", 14.154)
v_631 = RESULTS.get("QSE_pi_pi_star_eV", 10.580)
names = ["STO-3G\n(4 qubits)", "6-31G\n(4 qubits)", f"{BEST}\n(4 qubits)", "experiment"]
vals  = [v_sto, v_631, gap_qse_b, EXP_PI_PI_STAR]
cols  = ["#c0504d", "#e08a3c", "#3b6fb5", "0.55"]

fig, ax = plt.subplots(figsize=(9.5, 5.8))
bars = ax.bar(names, vals, color=cols, edgecolor="black", linewidth=0.8, width=0.55)
ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=11)
ax.axhline(EXP_PI_PI_STAR, ls="--", lw=1.5, color="green")
ax.annotate("", xy=(2, vals[2]), xytext=(0, vals[0]),
            arrowprops=dict(arrowstyle="->", color="0.3", lw=2,
                            connectionstyle="arc3,rad=-0.25"))
ax.text(1.0, max(vals) * 1.05,
        f"{vals[0] - vals[2]:.1f} eV gained\nfor 0 extra qubits",
        ha="center", fontsize=10, color="0.2",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.75"))
ax.set_ylabel(r"singlet $\pi \to \pi^*$ excitation energy (eV)", fontsize=11)
ax.set_title("VQE + QSE: the same 4 qubits, three basis sets", fontsize=12)
ax.set_ylim(0, max(vals) * 1.22)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

RESULTS["QSE_pi_pi_star_improved_eV"] = float(gap_qse_b)
RESULTS["improved_basis"] = BEST
