from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute, transpile
from VectorAmplitudeEncoder import VectorAmplitudeEncoder
from qiskit_aqt_provider.primitives import AQTSampler
from qiskit.primitives import Sampler
from qiskit_ibm_runtime import Options
import numpy as np

class CSWAPCircuit:
    def __init__(self, backend, shots):
        self._backend = backend
        self._shots = shots

    def _init_cswap_circuit(self, a, b):
        q1 = QuantumRegister(1, name='control_0')
        q2 = QuantumRegister(1, name='qr_1')
        q3 = QuantumRegister(3, name='qr_2')
        c = ClassicalRegister(1, name='c')
        qc = QuantumCircuit(q1,q2,q3,c)

        qc.prepare_state(a, q2[0:1])
        qc.prepare_state(b, q3[0:3])
        qc.barrier()
        qc.h(q1[0])
        qc.cswap(q1[0], q2[0], q3[0])
        qc.h(q1[0])
        qc.barrier()
        qc.measure(q1, c)

        return transpile(qc, self._backend, optimization_level=3)


    def run_cswap_circuit(self, qc, norm, noise_model):
        #job = execute(qc, self._backend, shots=self._shots, optimization_level=3, noise_model=None)
        options = Options()
        # options.transpilation.skip_transpilation = True
        options.execution.shots = 10000
        options.optimization_level = 3
        options.resilience_level = 3

        sampler = AQTSampler(self._backend)
        #sampler = Sampler()
        result = sampler.run(qc, shots=self._shots).result()
        #result = job.result()
        countsqd = result.quasi_dists[0]
        qdsquared = abs((4 * norm ** 2 * ((countsqd[0] / self._shots) - 0.5)))
        qd = np.sqrt(qdsquared)
        return qd

    def execute_swap_test(self, A, B, noise_model=None):
        cols, qd, rows, encoded_atoms = self.encode_atom_segments(A, B, VectorAmplitudeEncoder)
        for i in range(rows):
            for j in range(cols):
                norm_factor = encoded_atoms.norm_factor(A,B)[i]
                qc = self._init_cswap_circuit(
                    encoded_atoms.phi_reg(A, B)[i],
                    encoded_atoms.psi_reg(A, B)[j],
                )
                qd[i][j] = self.run_cswap_circuit(qc, norm_factor, noise_model)

        return qd

    def encode_atom_segments(self, A, B, encoder):
        encoded_atoms = encoder(A, B)
        rows = len(encoded_atoms.psi_reg(A, B))
        cols = len(encoded_atoms.psi_reg(A, B))
        qd = [0] * rows * cols
        qd = np.reshape(qd, (8, 8))
        return cols, qd, rows, encoded_atoms
