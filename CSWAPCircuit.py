from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute, transpile
from VectorAmplitudeEncoder import VectorAmplitudeEncoder
from qiskit_aqt_provider.primitives import AQTSampler
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
        #qc.barrier()
        qc.h(q1[0])
        #qc.cswap(q1[0], q2[0], q3[0])
        qc.cx(q2[0], q3[0])
        qc.cx(q1[0], q2[0])
        qc.cx(q2[0], q3[0])
        qc.cx(q1[0], q2[0])
        qc.cx(q2[0], q3[0])
        qc.h(q1[0])
        #qc.barrier()
        qc.measure(q1, c)

        return transpile(qc, self._backend, optimization_level=1)


    def run_cswap_circuit(self, qc, norm, noise_model):
        #job = execute(qc, self._backend, shots=self._shots, optimization_level=3, noise_model=None)
        sampler = AQTSampler(self._backend)
        result = sampler.run(qc).result()
        #result = job.result()
        countsqd = result.quasi_dists[0]
        qdsquared = abs((4 * norm ** 2 * ((countsqd[0] / self._shots) - 0.5)))
        qd = np.sqrt(qdsquared)
        return qd

    def execute_swap_test(self, A, B, noise_model=None):
        vector_preprocessor = VectorAmplitudeEncoder(A,B)
        rows = len(vector_preprocessor.psi_reg(A,B))
        cols = len(vector_preprocessor.psi_reg(A,B))
        qd = [0] * rows * cols
        qd = np.reshape(qd, (8,8))
        for i in range(rows):
            for j in range(cols):
                norm_factor = vector_preprocessor.norm_factor(A,B)[i]
                qc = self._init_cswap_circuit(
                    vector_preprocessor.phi_reg(A, B)[i],
                    vector_preprocessor.psi_reg(A, B)[j],
                )

            qd[i][j] = self.run_cswap_circuit(qc, norm_factor, noise_model)

        return qd

        """
        return [self._init_cswap_circuit(vector_preprocessor.phi_reg(A,B)[i],
                                  vector_preprocessor.psi_reg(A,B)[i],
                                  vector_preprocessor.norm_factor(A,B)[i],
                                         noise_model) for i in range(len(vector_preprocessor.psi_reg(A,B)))]
        """



