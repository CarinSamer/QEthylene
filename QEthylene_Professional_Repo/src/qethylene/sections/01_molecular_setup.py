def ethylene_geometry(twist_deg=0.0, r_cc=1.339, r_ch=1.086, hch_deg=117.4):
    half = np.radians(hch_deg) / 2.0
    cx = r_cc / 2.0
    hx, hy = r_ch * np.cos(half), r_ch * np.sin(half)
    atoms = [("C", (-cx, 0, 0)), ("C", (cx, 0, 0)),
             ("H", (-cx - hx,  hy, 0)), ("H", (-cx - hx, -hy, 0))]
    t = np.radians(twist_deg)
    R = np.array([[1, 0, 0],
                  [0, np.cos(t), -np.sin(t)],
                  [0, np.sin(t),  np.cos(t)]])
    for s in (+1, -1):
        p = R @ np.array([hx, s * hy, 0.0])
        atoms.append(("H", (cx + p[0], p[1], p[2])))
    return [f"{el} {x:.6f} {y:.6f} {z:.6f}" for el, (x, y, z) in atoms]

print("Planar ethylene (0 deg):")
print("\n".join(ethylene_geometry(0)))
print("\nTwisted ethylene (90 deg) -- the last two H atoms rotate out of the plane:")
print("\n".join(ethylene_geometry(90)))

def build_problem(twist_deg=0.0, basis="sto3g", n_elec=2, n_orb=2, verbose=True):
    driver = PySCFDriver(atom=ethylene_geometry(twist_deg), basis=basis,
                         charge=0, spin=0, unit=DistanceUnit.ANGSTROM)
    full = driver.run()
    n_occ = full.num_particles[0]                       # doubly-occupied Molecular Orbitals (8)
    homo, lumo = n_occ - 1, n_occ
    # n_elec electrons fill n_elec//2 doubly-occupied active orbitals; every
    # remaining active orbital comes from the virtuals. This is what makes
    # CAS(2,4) select the pi orbital plus three virtuals, as intended.
    n_act_occ = n_elec // 2
    n_act_vir = n_orb - n_act_occ
    active = list(range(homo - n_act_occ + 1, homo + 1)) + \
             list(range(lumo, lumo + n_act_vir))
    ast = ActiveSpaceTransformer(num_electrons=n_elec, num_spatial_orbitals=n_orb,
                                 active_orbitals=active)
    cas = ast.transform(full)
    if verbose:
        try:
            print(f"  Hartree-Fock (mean-field) total energy = {full.reference_energy:.8f} Ha")
        except AttributeError:
            pass                      # older drivers lack reference_energy
        print(f"  full molecule: {full.num_particles} electrons, "
              f"{full.num_spatial_orbitals} spatial orbitals")
        print(f"  active orbitals (0-indexed): {active}   -> HOMO={homo}, LUMO={lumo}")
        print(f"  active space: {cas.num_particles} electrons, "
              f"{cas.num_spatial_orbitals} spatial orbitals")
    return full, cas, active

print("=== Planar (0 deg) ===")
full_eq, cas_eq, active_eq = build_problem(0.0)
print("\nMolecular orbital energies (Ha) -- HOMO/LUMO straddle the sign change:")
for i, e in enumerate(full_eq.orbital_energies):
    tag = "  <- HOMO (pi)" if i == active_eq[0] else ("  <- LUMO (pi*)" if i == active_eq[-1] else "")
    print(f"  MO {i:2d}: {e:+.6f}{tag}")

jw = JordanWignerMapper()

H_ferm_eq = cas_eq.hamiltonian.second_q_op()
H_jw_eq = jw.map(H_ferm_eq)

print("Jordan-Wigner qubit Hamiltonian (planar):")
print(f"  qubits      = {H_jw_eq.num_qubits}")
print(f"  Pauli terms = {len(H_jw_eq)}")
print(H_jw_eq)

par = ParityMapper(num_particles=cas_eq.num_particles)
H_par_eq = par.map(H_ferm_eq)

# The two-qubit reduction fixes the alpha parity and the total parity, which for
# two spatial orbitals pins the state to n_alpha = n_beta = 1: the N=2, Sz=0
# sector, four states. So the 4 eigenvalues of the tapered Hamiltonian must
# reproduce the 4 eigenvalues of the 4-qubit Hamiltonian in that sector -- every
# excited state, not merely the minimum. N and Sz are diagonal in the
# computational basis, so the sector can be read off the bitstrings directly.
_N  = np.array([bin(k).count("1") for k in range(16)])
_Sz = np.array([0.5 * (((k >> 0) & 1) + ((k >> 1) & 1) - ((k >> 2) & 1) - ((k >> 3) & 1))
                for k in range(16)])
_ev, _vc = np.linalg.eigh(H_jw_eq.to_matrix())
_w = np.abs(_vc) ** 2
_in_sector = [k for k in range(16)
              if abs(_w[:, k] @ _N - 2) < 1e-6 and abs(_w[:, k] @ _Sz) < 1e-6]
sector_jw = np.sort(_ev[_in_sector])
ev_par    = np.sort(np.linalg.eigvalsh(H_par_eq.to_matrix()))

print(f"parity Hamiltonian: {H_par_eq.num_qubits} qubits, {len(H_par_eq)} terms")
assert len(sector_jw) == len(ev_par), (
    f"sector holds {len(sector_jw)} states but tapered H has {len(ev_par)} -- "
    "the tapering projected onto the wrong symmetry sector")

print(f"\n{'root':>5}{'4-qubit JW (N=2, Sz=0)':>26}{'2-qubit parity':>19}{'diff (ulp)':>12}")
print("-" * 62)
ulps = []
for k, (a, b) in enumerate(zip(sector_jw, ev_par)):
    u = abs(a - b) / np.spacing(abs(a))
    ulps.append(u)
    print(f"{k:>5}{a:>26.12f}{b:>19.12f}{u:>12.1f}")
print(f"\nlargest disagreement anywhere in the spectrum = {max(ulps):.1f} ulp")
print("(1 ulp is the gap between adjacent float64 values, so a few ulp is exact")
print(" agreement: the two Hamiltonians are the same operator. A real error would")
print(" be millihartree -- billions of ulp -- and would usually hit an excited")
print(" root before the ground state.)")

E0_exact = sector_jw[0]
RESULTS["E0_exact_active_Ha"] = float(E0_exact)
RESULTS["taper_max_ulp"] = float(max(ulps))
