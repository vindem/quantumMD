from qiskit.algorithms.minimum_eigen_solvers import VQE
import numpy as np
from qiskit.opflow import MatrixOp
from CSWAPCircuit import CSWAPCircuit

def calculate_distance_quantum(A, B, backend, noise_model):
    cswap_circuit = CSWAPCircuit(1, 1, 3, 1, backend, 8192)
    quantum_ED = cswap_circuit.execute_swap_test(A, B, noise_model)
    return quantum_ED


def calc_eigval_quantum(bpm, ansatz, optimizer, quantum_instance):
    qubit_op = MatrixOp(-bpm)
    variational_form = ansatz
    vqe = VQE(qubit_op, variational_form, optimizer=optimizer)
    ret = vqe.run(quantum_instance)
    vqe_result = np.real(ret['eigenvalue'])
    return vqe_result