import os
os.environ["QCS_SETTINGS_APPLICATIONS_QVM_URL"] = "http://127.0.0.1:5001"
os.environ["QCS_SETTINGS_APPLICATIONS_QUILC_URL"] = "tcp://127.0.0.1:5002"

import numpy as np
import matplotlib.pyplot as plt

# from scipy.optimize import minimize
from pyquil import Program, get_qc
from pyquil.api import WavefunctionSimulator
from pyquil.paulis import sZ, PauliSum
from pyquil.gates import RX, MEASURE
from pyquil.quilbase import Declare

# Wavefunction simulator for exact expectation values
wfn_sim = WavefunctionSimulator()


# Ansatz: single RX rotation on qubit 0
def small_ansatz(params):
    return Program(RX(params[0], 0))


# Print ansatz with example parameter
print("Ansatz with example value for parameter:")
print(small_ansatz([1.0]))

# Hamiltonian H = Z_0
hamiltonian = PauliSum([sZ(0)])


# Exact expectation via wavefunction simulator
def exact_expectation(angle):
    return wfn_sim.expectation(small_ansatz([angle]), hamiltonian).real


# Sampled expectation by measuring Z manually
qc = get_qc("1q-qvm")


def sampled_expectation(angle, n_samples=1000):
    p = Program(
        Declare("ro", "BIT", 1),
        RX(angle, 0),
        MEASURE(0, ("ro", 0)),
    ).wrap_in_numshots_loop(n_samples)
    result = qc.run(qc.compile(p)).get_register_map()["ro"]
    # Z eigenvalues: 0 -> +1, 1 -> -1
    return 1 - 2 * np.mean(result)


# Check at a single angle
angle = 2.0
print("Exact expectation at angle = {}:".format(angle))
print(exact_expectation(angle))

# Angle range
angle_range = np.linspace(0.0, 2.0 * np.pi, 20)

# Exact values
exact = [exact_expectation(angle) for angle in angle_range]
plt.plot(angle_range, exact, linewidth=2, label="Exact")

# Sampled values
sampled = [sampled_expectation(angle, n_samples=1000) for angle in angle_range]
plt.plot(angle_range, sampled, "-o", label="Sampled")

# Plot
plt.xlabel("Angle [radians]")
plt.ylabel("Expectation value")
plt.legend()
plt.grid()
plt.show()