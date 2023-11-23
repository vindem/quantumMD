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

from qiskit.transpiler import PassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime.transpiler.passes.scheduling import ALAPScheduleAnalysis, PadDynamicalDecoupling
from qiskit.circuit.library import XGate

callback_dict = {
    "prev_vector": None,
    "iters": 0,
    "cost_history": [],
}

def build_callback(ansatz, hamiltonian, estimator, callback_dict):
    """Return callback function that uses Estimator instance,
    and stores intermediate values into a dictionary.

    Parameters:
        ansatz (QuantumCircuit): Parameterized ansatz circuit
        hamiltonian (SparsePauliOp): Operator representation of Hamiltonian
        estimator (Estimator): Estimator primitive instance
        callback_dict (dict): Mutable dict for storing values

    Returns:
        Callable: Callback function object
    """

    def callback(current_vector):
        """Callback function storing previous solution vector,
        computing the intermediate cost value, and displaying number
        of completed iterations and average time per iteration.

        Values are stored in pre-defined 'callback_dict' dictionary.

        Parameters:
            current_vector (ndarray): Current vector of parameters
                                      returned by optimizer
        """
        # Keep track of the number of iterations
        callback_dict["iters"] += 1
        # Set the prev_vector to the latest one
        callback_dict["prev_vector"] = current_vector
        # Compute the value of the cost function at the current vector
        # This adds an additional function evaluation
        current_cost = (
            estimator.run(ansatz, hamiltonian, parameter_values=current_vector).result().values[0]
        )
        callback_dict["cost_history"].append(current_cost)
        # Print to screen on single line
        print(
            "Iters. done: {} [Current cost: {}]".format(callback_dict["iters"], current_cost),
            end="\r",
            flush=True,
        )

    return callback

def calculate_distance_quantum(A, B):
    #IBMQ.load_account()
    #provider = IBMQ.get_provider(hub='ibm-q-research-2', group='vienna-uni-tech-1', project='main')
    #backend = Aer.get_backend('aer_simulator')
    #noise_model = NoiseModel.from_backend(provider.backend.ibmq_quito)
    provider = AQTProvider("i_che_token")
    backend = provider.get_backend("offline_simulator_no_noise")
    cswap_circuit = CSWAPCircuit(1, 1, 3, 1, backend, 200)
    quantum_ED = cswap_circuit.execute_swap_test(A, B)
    arr = np.array(quantum_ED)
    #print(arr.shape)
    return arr


def calc_eigval_quantum(bpm, ansatz, backend, optimizer):
    qubit_op = Operator(-bpm)
    #backend = Aer.get_backend('aer_simulator')
    provider = AQTProvider("i_che_token")
    backend = provider.get_backend("offline_simulator_no_noise")
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
    """
    pm = generate_preset_pass_manager(target=backend, optimization_level=3)
    pm.scheduling = PassManager(
        [
            ALAPScheduleAnalysis(durations=backend.durations()),
            PadDynamicalDecoupling(
                durations=backend.durations(),
                dd_sequences=[XGate(),XGate()],
                pulse_alignment=backend.pulse_alignment
            )
        ])
    
    improved_ansatz = pm.run(ansatz)
    """
    improved_ansatz = ansatz
    num_params = improved_ansatz.num_parameters
    #x0 = (np.max(qubit_op) - np.min(qubit_op)) * np.random.random(num_params)
    #x0 = np.max(np.real(qubit_op)) * np.random.random(num_params)

    x0 = 2 * np.pi * np.random.random(num_params) * 10.0
    print("x0" + str(x0))
    options = Options()
    #options.transpilation.skip_transpilation = True
    options.execution.shots = 10000
    options.optimization_level = 3
    options.resilience_level = 3

    with Session(service=service, backend=backend):
        #estimator = Estimator(options=options)
        #callback = build_callback(improved_ansatz, qubit_op, estimator, callback_dict)
        estimator = AQTEstimator(backend=backend, options={"optimization_level":3, "resilience_level": 3})
        start = time.time()
        res = minimize(
            cost_function,
            x0,
            args=(improved_ansatz, qubit_op, estimator),
            method="cobyla",
            options={"maxiter":1000}
        )
        end = time.time()
        print("FINAL: "+ str(np.real(res['x'])))
        return [max(np.real(res['x'])), end - start]
