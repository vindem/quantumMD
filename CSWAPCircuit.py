from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute, transpile
from VectorAmplitudeEncoder import VectorAmplitudeEncoder
import numpy as np

class CSWAPCircuit:
    def __init__(self, aux, qr1n, qr2n, crn, backend, shots):
        self._aux = aux
        self._qr1n = qr1n
        self._qr2n = qr2n
        self._crn = crn
        self._backend = backend
        self._shots = shots

    def _init_cswap_circuit(self, a, b):
        q1 = QuantumRegister(self._aux, name='control_0')
        q2 = QuantumRegister(self._qr1n, name='qr_1')
        q3 = QuantumRegister(self._qr2n, name='qr_2')
        c = ClassicalRegister(self._crn, name='c')
        qc = QuantumCircuit(q1,q2,q3,c)

        qc.initialize(a, q2[0:self._qr1n])
        qc.initialize(b, q3[0:self._qr2n])
        qc.barrier()
        qc.h(q1[0])
        qc.cswap(q1[0], q2[0], q3[0])
        qc.h(q1[0])
        qc.barrier()
        qc.measure(q1, c)

        return qc

    def run_cswap_circuit(self, qc, norm, noise_model):
        job = execute(qc, self._backend, shots=self._shots, optimization_level=3, noise_model=noise_model)
        result = job.result()
        countsqd = result.get_counts(qc)
        qdsquared = abs((4 * norm ** 2 * ((countsqd['0'] / self._shots) - 0.5)))
        qd = np.sqrt(qdsquared)
        return qd

    def execute_swap_test(self, A, B, noise_model):
        vector_preprocessor = VectorAmplitudeEncoder(A,B)
        qd = [0] * len(vector_preprocessor.psi_reg(A,B))
        for i in range(len(vector_preprocessor.psi_reg(A,B))):
            norm_factor = vector_preprocessor.norm_factor(A,B)[i]
            qc = self._init_cswap_circuit(
                vector_preprocessor.phi_reg(A, B)[i],
                vector_preprocessor.psi_reg(A, B)[i],
            )
            qd[i] = self.run_cswap_circuit(qc, norm_factor, noise_model)

        return qd

        """
        return [self._init_cswap_circuit(vector_preprocessor.phi_reg(A,B)[i],
                                  vector_preprocessor.psi_reg(A,B)[i],
                                  vector_preprocessor.norm_factor(A,B)[i],
                                         noise_model) for i in range(len(vector_preprocessor.psi_reg(A,B)))]
        """



