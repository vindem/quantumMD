from qiskit.algorithms.minimum_eigen_solvers import VQE
import numpy as np
from qiskit.opflow import MatrixOp
from CSWAPCircuit import CSWAPCircuit
from qiskit.providers import JobStatus
from qiskit import IBMQ
import time
from qiskit import IBMQ, Aer
from qiskit.providers.aer.noise import NoiseModel


def calculate_distance_quantum(A, B):
    IBMQ.load_account()
    provider = IBMQ.get_provider(hub='ibm-q-research-2', group='vienna-uni-tech-1', project='main')
    backend = Aer.get_backend('aer_simulator')
    noise_model = NoiseModel.from_backend(provider.backend.ibmq_quito)
    cswap_circuit = CSWAPCircuit(1, 1, 3, 1, backend, 8192)
    quantum_ED = cswap_circuit.execute_swap_test(A, B, noise_model)
    arr = np.array(quantum_ED)
    print(arr.shape)
    return arr


def calc_eigval_quantum(bpm, ansatz, backend, optimizer):
    qubit_op = MatrixOp(-bpm)

    #vqe = VQE(qubit_op, variational_form, optimizer=optimizer)

    vqe_inputs = {
        'ansatz': ansatz,
        'operator': qubit_op,
        'optimizer': optimizer,
        'initial_point': np.random.random(ansatz.num_parameters),
        'measurement_error_mitigation': True,
        'shots': 1024
        }
    options = {
        'backend_name': backend,
    }
    provider = IBMQ.get_provider('ibm-q-research-2', 'vienna-uni-tech-1', 'main')
    job = provider.runtime.run(program_id='vqe',
                                     inputs=vqe_inputs,
                                     options=options)

    while job.status() != JobStatus.RUNNING and job.status() != JobStatus.ERROR:
        pass
    if job.status() == JobStatus.ERROR:
        print(job.error_message())
        print(job.logs())

    start = time.time()
    res = job.result()
    end = time.time()
    return [-np.real(res['eigenvalue']), end - start]
