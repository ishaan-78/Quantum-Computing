# !pip -q install qiskit qiskit-aer

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace, SparsePauliOp
from qiskit_aer import AerSimulator
from itertools import product
from qiskit.visualization import plot_bloch_vector
import matplotlib.pyplot as plt

def analyze(qc, title=None):
    qc_copy = qc.copy()
    ops = [(ci.operation.name, [qc_copy.find_bit(q).index for q in ci.qubits])
          for ci in qc_copy.data
          if ci.operation.name not in ("measure", "barrier")]
    step = len(ops)
    prev = f"{ops[-1][0]} on {ops[-1][1]}" if step else "None"
    print(f"\n[Step {step}] {f'{title} — ' if title else ''}prev: {prev}")


    sv = Statevector.from_instruction(qc_copy)
    n = sv.num_qubits
    amplitudes = sv.data
    probabilities = np.abs(amplitudes)**2
    basis = [format(i, f"0{qc_copy.num_qubits}b") for i in range(2**qc_copy.num_qubits)]

    print("Statevector probabilities (no measurement):")
    for b, p in zip(basis, probabilities):
        if p > 1e-6:
            print(f"  |{b}⟩: {p:.4f}")

    if(n != 1):
      axes = ('X','Y','Z'); eps = 1e-3
      print("Pairwise correlations (showing non-zero among 9 per pair):")
      for i in range(n):
        for j in range(i+1, n):
          shown = []
          for a, b in product(axes, repeat=2):
            lab = ['I'] * n
            lab[n-1-i] = a
            lab[n-1-j] = b
            op = SparsePauliOp.from_list([(''.join(lab), 1.0)])
            val = float(np.real(sv.expectation_value(op)))
            if abs(val) >= eps:
              shown.append(f"{a}{b}={val:+.3f}")
          if shown:
            print(f"  q{i}⊗q{j}: " + ", ".join(shown))

    else:
      dm = DensityMatrix(sv)
      X = np.array([[0,1],[1,0]], complex)
      Y = np.array([[0,-1j],[1j,0]], complex)
      Z = np.array([[1,0],[0,-1]], complex)

      print("Bloch ⟨X,Y,Z⟩ per qubit:")
      nq = qc_copy.num_qubits
      for q in range(nq):
        trace_out = [i for i in range(nq) if i != q]
        rho = partial_trace(dm, trace_out).data
        rx = float(np.real(np.trace(rho @ X)))
        ry = float(np.real(np.trace(rho @ Y)))
        rz = float(np.real(np.trace(rho @ Z)))
        print(f"  q{q}: ({rx:.3f}, {ry:.3f}, {rz:.3f})")
        bloch = [rx, ry, rz]
        fig = plot_bloch_vector(bloch, title=f"q{q} Bloch sphere")
        # display(fig)

n_qubits = 1
shots = 10000

circuit = QuantumCircuit(n_qubits)

# --------------TODO: Add Gates--------------------
# Use circuit.h(0), circuit.x(0), circuit.y(0), circuit.z(0)
circuit.h(0)
analyze(circuit)
circuit.barrier()
circuit.h(0)
analyze(circuit)
# -------------------------------------------------

print("\nFinal Circuit:")
print(circuit.draw())

sim = AerSimulator()
qc_meas = circuit.copy()
qc_meas.measure_all()
res = sim.run(transpile(qc_meas, sim), shots=shots).result().get_counts()
total = sum(res.values())
probs_emp = {k: v/total for k, v in sorted(res.items())}

print("\nMeasurement probabilities (10,000 shots):")
for b, p in probs_emp.items():
    print(f"  |{b}⟩: {p:.4f}")