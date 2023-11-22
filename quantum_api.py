from qiskit.algorithms.minimum_eigen_solvers import VQE
import numpy as np
from qiskit.opflow import MatrixOp
from CSWAPCircuit import CSWAPCircuit
from qiskit.providers import JobStatus
from qiskit import IBMQ
import time
from qiskit import IBMQ, Aer
from qiskit_ibm_runtime import QiskitRuntimeService, Estimator, Session, Options
from qiskit.providers.aer.noise import NoiseModel
from qiskit_aqt_provider import AQTProvider
from qiskit_aqt_provider.primitives import AQTEstimator
from qiskit.quantum_info import Operator
from scipy.optimize import minimize


def calculate_distance_quantum(A, B):
    IBMQ.load_account()
    provider = IBMQ.get_provider(hub='ibm-q-research-2', group='vienna-uni-tech-1', project='main')
    backend = Aer.get_backend('aer_simulator')
    #noise_model = NoiseModel.from_backend(provider.backend.ibmq_quito)
    #provider = AQTProvider("i_che_token")
    #backend = provider.get_backend("offline_simulator_no_noise")
    cswap_circuit = CSWAPCircuit(1, 1, 3, 1, backend, 200)
    quantum_ED = cswap_circuit.execute_swap_test(A, B)
    arr = np.array(quantum_ED)
    #print(arr.shape)
    return arr


def calc_eigval_quantum(bpm, ansatz, backend, optimizer):
    qubit_op = Operator(-bpm)
    #provider = AQTProvider("i_che_token")
    #backend = provider.get_backend("offline_simulator_no_noise")
    #vqe = VQE(qubit_op, variational_form, optimizer=optimizer)
    """
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
    """
    service = QiskitRuntimeService()
    def cost_function(params, ansatz, hamiltonian, estimator):
        energy = estimator.run(ansatz, hamiltonian, parameter_values=params).result().values[0]
        return energy


    num_params = ansatz.num_parameters
    x0 = 2 * np.pi * np.random.random(num_params)

    with Session(service=service, backend=backend):
        estimator = Estimator()
        #estimator = AQTEstimator(backend=backend)
        start = time.time()
        res = minimize(
            cost_function,
            x0,
            args=(ansatz, qubit_op, estimator),
            method="cobyla"
        )
        end = time.time()
        print(res)
        return [-np.real(res['x']), end - start]
