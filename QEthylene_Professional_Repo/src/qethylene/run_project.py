"""Execute the extracted notebook sections in their original order.

This runner is provided for users who choose to execute the project.
It is NOT invoked by the repository CI workflow.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SECTIONS_DIR = Path(__file__).resolve().parent / "sections"


def run():
    namespace = {"__name__": "__main__"}
    section_files = [
        "00_setup.py",
        "01_molecular_setup.py",
        "02_ground_state_vqe.py",
        "03_excited_states_qse.py",
        "04_noise_aware_benchmarking.py",
        "05_classical_benchmarking.py",
        "06_marcus_extension.py",
        "07_improved_basis.py",
        "08_sample_based_diagonalization.py",
        "09_headline_results.py",
    ]
    for filename in section_files:
        path = SECTIONS_DIR / filename
        source = path.read_text(encoding="utf-8")
        print(f"\\n===== {filename} =====")
        exec(compile(source, str(path), "exec"), namespace)
    return namespace


if __name__ == "__main__":
    run()
