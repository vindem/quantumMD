from qiskit import IBMQ, Aer
from CSWAPCircuit import CSWAPCircuit
import random
from scipy.spatial import distance
from sklearn.metrics import mean_squared_error

random.seed(42)

IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q-research-2', group='vienna-uni-tech-1', project='main')
backend_sim = provider.backend.ibmq_qasm_simulator

def generate_random_vector(n_points):
    vec = []
    for i in range(n_points):
        vec.append([random.uniform(0, 1) for j in range(3)])
    return vec

backend = Aer.get_backend('qasm_simulator')

for num_qubits in range(0,5):
    quantum_distances = []
    classic_distances = []
    for i in range(10):
        A = generate_random_vector(2**num_qubits)
        B = generate_random_vector(2**num_qubits)

        cswap_circuit = CSWAPCircuit(1, 1, 3, 1, backend_sim, 8192)
        quantum_ED = cswap_circuit.execute_swap_test(A, B)
        classic_ED = [distance.euclidean(A[i], B[i]) for i in range(len(A))]

        quantum_distances.append(quantum_ED)
        classic_distances.append(classic_ED)
    mse = mean_squared_error(quantum_distances, classic_distances)
    print(mse)





