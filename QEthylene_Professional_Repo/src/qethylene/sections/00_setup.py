import warnings
# Silence library deprecation chatter only -- numerical warnings
# (overflow, invalid value, non-convergence) still come through.
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import qiskit, qiskit_nature, qiskit_aer
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, DensityMatrix, SparsePauliOp, Operator

from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.operators import FermionicOp

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError

from pyscf import gto, scf, mcscf          # classical references

HARTREE2EV = 27.211386245988          # 1 Hartree in electron-volts
EXP_PI_PI_STAR = 7.6                    # experimental ethylene pi->pi* band max (eV)
np.set_printoptions(precision=6, suppress=True)
RESULTS = {}                            # headline numbers collected here

print("qiskit        ", qiskit.__version__)
print("qiskit-nature ", qiskit_nature.__version__)
print("qiskit-aer    ", qiskit_aer.__version__)
