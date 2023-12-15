from qiskit.algorithms.minimum_eigen_solvers import VQE
import numpy as np
from qiskit.opflow import MatrixOp
from CSWAPCircuit import CSWAPCircuit
from qiskit.providers import JobStatus
from qiskit import IBMQ
import time
from qiskit import IBMQ, Aer
from qiskit_ibm_runtime import QiskitRuntimeService, Estimator, Session, Options
from qiskit.primitives import Estimator
from qiskit.providers.aer.noise import NoiseModel
from qiskit_aqt_provider import AQTProvider
from qiskit_aqt_provider.primitives import AQTEstimator
from qiskit.quantum_info import Operator
from scipy.optimize import minimize
from config import Config
from qiskit.algorithms.optimizers import COBYLA
import math
from qiskit.transpiler import PassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime.transpiler.passes.scheduling import ALAPScheduleAnalysis, PadDynamicalDecoupling
from qiskit.circuit.library import XGate
from qiskit.circuit.library import *
from JobPersistenceManager import JobPersistenceManager

callback_dict = {
    "prev_vector": None,
    "iters": 0,
    "cost_history": [],
}

def setup_backend_for_task(task_name):
    task_setup = Config.execution_setup['execution_setup'][task_name][0]
    match task_setup['provider']:
        case 'Aer':
            return Aer.get_backend('aer_simulator')
        case 'AQT':
            provider = AQTProvider(task_setup['token'])
            backend = provider.get_backend(name=task_setup['name'])
            return backend
        case 'IBM':
            IBMQ.load_account()
            provider = IBMQ.get_provider(hub='ibm-q-research-2', group='vienna-uni-tech-1', project='main')
            backend = provider.get_backend(name=task_setup['name'])
            return backend
        case _:
            raise Exception("Unknown Provider!")


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
    print("Starting SWAPTEST.")
    distance_configuration = Config.execution_setup['execution_setup']['dist_calc'][0]
    print("Target machine: "+distance_configuration['name']+ ", " + distance_configuration['type'])
    #IBMQ.load_account()
    #provider = IBMQ.get_provider(hub='ibm-q-research-2', group='vienna-uni-tech-1', project='main')
    #backend = Aer.get_backend('aer_simulator')
    #noise_model = NoiseModel.from_backend(provider.backend.ibmq_quito)
    #backend = provider.get_backend("ibex", workspace="hpqc")
    backend = setup_backend_for_task('dist_calc')
    cswap_circuit = CSWAPCircuit(backend, 200)
    quantum_ED = cswap_circuit.execute_swap_test(A, B)
    arr = np.array(quantum_ED)
    #print(arr.shape)
    return arr


def calc_eigval_quantum(bpm, filename):
    print("Starting VQE.")
    eigenvalue_configuration = Config.execution_setup['execution_setup']['eigenvalues'][0]
    print("Target machine: " + eigenvalue_configuration['name'] + ", " + eigenvalue_configuration['type'])
    qubit_op = Operator(-bpm)
    #backend = Aer.get_backend('aer_simulator')
    backend = setup_backend_for_task('eigenvalues')
    #vqe = VQE(qubit_op, variational_form, optimizer=optimizer)

    def cost_function(params, ansatz, hamiltonian, estimator):
        try:
            if eigenvalue_configuration['persistence']:
                persistence_manager = JobPersistenceManager()
            job = estimator.run(ansatz, hamiltonian, parameter_values=params)
            if eigenvalue_configuration['persistence']:
                persistence_manager.add_id(job.job_id())
            result = job.result()
        except TimeoutError:
            if eigenvalue_configuration['persistence']:
                job_ids = persistence_manager.active_jobs()
                for job_id in job_ids:
                    restored_job = AQTJob.restore(job_id, access_token=eigenvalue_configuration['token'])
                    result = restored_job.result()

        energy = result.values[0]
        return energy



    improved_ansatz = globals()[eigenvalue_configuration['ansatz']](int(math.log2(bpm.shape[0])))
    num_params = improved_ansatz.num_parameters
    #x0 = - (np.max(qubit_op) - np.min(qubit_op)) * np.random.random(num_params) * 10.0
    #x0 = np.max(np.real(qubit_op)) * np.random.random(num_params)

    x0 = [0] * num_params
    #x0 = np.random.random(num_params)
    #print("x0" + str(x0))
    options = Options()
    #options.transpilation.skip_transpilation = True
    options.execution.shots = 200
    options.optimization_level = 3
    options.resilience_level = 3

    #estimator = Estimator(options=options)
    #callback = build_callback(improved_ansatz, qubit_op, estimator, callback_dict)
    if eigenvalue_configuration['provider'] == 'AQT':
        from qiskit_aqt_provider.aqt_job import AQTJob
        estimator = AQTEstimator(backend=backend)
    elif eigenvalue_configuration['provider'] == 'IBM':
        estimator = Estimator(options={"optimization_level":3, "resilience_level": 3})
    else:
        estimator = Estimator(options={"optimization_level": 3, "resilience_level": 3})
    start = time.time()
    res = minimize(
        cost_function,
        x0,
        args=(improved_ansatz, qubit_op, estimator),
        method=eigenvalue_configuration['optimizer'],
        options={"maxiter": eigenvalue_configuration['opt_iters'], "tol":0.1}
        )
    end = time.time()

    #print("FINAL: "+ str(np.real(-res.fun)))
    return [-(np.real(res.fun)), end - start]
