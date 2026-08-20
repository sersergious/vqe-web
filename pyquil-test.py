from dotenv import load_dotenv
load_dotenv()  # must be called before importing pyquil_for_azure_quantum

from pyquil_for_azure_quantum import get_qvm, get_qpu
from pyquil.gates import H, CNOT, MEASURE
from pyquil.quil import Program
from pyquil.quilbase import Declare

# Build a Bell state program
program = Program(
    Declare("ro", "BIT", 2),
    H(0),
    CNOT(0, 1),
    MEASURE(0, ("ro", 0)),
    MEASURE(1, ("ro", 1)),
).wrap_in_numshots_loop(100)

# Use the QVM simulator (free)
qvm = get_qvm()
exe = qvm.compile(program)
results = qvm.run(exe)

print("Results:")
print(results.readout_data["ro"])
