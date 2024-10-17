from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile
from VectorAmplitudeEncoder import VectorAmplitudeEncoder
from qiskit_aqt_provider.primitives import AQTSampler
from qiskit_aqt_provider.aqt_job import AQTJob
from qiskit.primitives import Sampler
from qiskit_ibm_runtime import Options
import numpy as np
from config import Config
from JobPersistenceManager import JobPersistenceManager
import quantum_api

job_ids = set()
class CSWAPCircuit:
    def __init__(self, backend, shots):
        self._backend = backend
        self._shots = shots
        self._show_figure = True

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

        return qc


    def run_cswap_circuit(self, qc, norm, noise_model):
        #job = execute(qc, self._backend, shots=self._shots, optimization_level=3, noise_model=None)

        options = Options()
        # options.transpilation.skip_transpilation = True
        options.execution.shots = 1024
        options.optimization_level = 3
        options.resilience_level = 3
        distance_configuration = Config.execution_setup['execution_setup']['dist_calc'][0]
        if distance_configuration['provider'] == 'AQT':
            sampler = AQTSampler(self._backend)
        else:
            sampler = Sampler()
        try:
            job = sampler.run(qc, shots=self._shots)
            if distance_configuration['persistence']:
                persistence_manager = JobPersistenceManager()
                persistence_manager.add_id(job.job_id())
            result = job.result()
            if distance_configuration['persistence']:
                persistence_manager.remove_id(job.job_id())
        #result = job.result()
        except TimeoutError:
            if distance_configuration['persistence']:
                job_ids = persistence_manager.active_jobs()
                for job_id in job_ids:
                    restored_job = AQTJob.restore(job_id, access_token=distance_configuration['token'])
                    result = restored_job.result()
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
        qd = np.reshape(qd, (rows, cols))
        return cols, qd, rows, encoded_atoms
