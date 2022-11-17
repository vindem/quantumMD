from qiskit import IBMQ
import numpy as np
import scipy.linalg as lin_alg
from qiskit.circuit.library import RealAmplitudes, EfficientSU2, TwoLocal, PauliTwoDesign, ExcitationPreserving
from qiskit.opflow import MatrixOp, X, Y, Z, I, PauliOp, PrimitiveOp
from qiskit.providers.ibmq.runtime.exceptions import RuntimeJobFailureError
from qiskit.quantum_info import Operator, Pauli
from qiskit.algorithms.optimizers import COBYLA, SPSA, GradientDescent
from qiskit.providers.ibmq import least_busy
from qiskit.providers import JobStatus
from itertools import product
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import csv
from qiskit import QuantumCircuit, transpile
from quantum_utils import pauli_representation

provider = None

def test_quantum_eigenvalue(q_shots = 8192, opt_iter = 100):
    classic = []
    quantum = []
    opt_times = []
    tot_times = []

    IBMQ.load_account()
    provider = IBMQ.get_provider(hub='ibm-q-research-2', group='vienna-uni-tech-1', project='main')

    backend_sim = provider.backend.ibmq_qasm_simulator
    backend_real = provider.get_backend("ibm_perth")
    coupling_map = backend_real.configuration().coupling_map
    # Extracting noise model for error mitigation
    noise_model = NoiseModel.from_backend(backend_real)
    quantum_instance = QuantumInstance(backend=backend_sim,
                                       shots=q_shots,
                                       noise_model=noise_model,
                                       coupling_map=coupling_map,
                                       measurement_error_mitigation_cls=CompleteMeasFitter,
                                       cals_matrix_refresh_period=30)
    optimizer = COBYLA(maxiter=opt_iter, tol=0.0001)
    num_qubits = 2
    # definition of the ansatz
    best_ansatz_ever = RealAmplitudes(num_qubits, reps=2)

    dimensions = 2
    for i in range(1,20):
        print('Iteration '+str(i))
        rdist = np.random.rand(dimensions,dimensions)
        rdistt = np.transpose(rdist)
        z1 = np.zeros((dimensions, dimensions))
        z2 = np.zeros((dimensions, dimensions))
        in_matrix = np.block([[z1, rdist], [rdistt, z2]])
        #print(in_matrix)

        #hamiltonian_qubit_op = MatrixOp(in_matrix)
        #print(hamiltonian_qubit_op.num_qubits)
        #hamiltonian = FermionicOperator(h1=in_matrix)
        #hamiltonian_qubit_op = hamiltonian.mapping(map_type='parity')
        hamiltonian = MatrixOperator(in_matrix)
        hamiltonian_qubit_op = op_converter.to_weighted_pauli_operator(hamiltonian)

        vqe = VQE(operator=hamiltonian_qubit_op,var_form=best_ansatz_ever,quantum_instance=quantum_instance, optimizer=optimizer)

        start = time.time()
        vqe_result = np.real(vqe.run(backend_real)['eigenvalue'])
        end = time.time()

        tot_time = end-start
        print("VQE eigenvalues: " + str(vqe_result))
        print("MD eigenvalue: " + str(lin_alg.eigvalsh(in_matrix)))

        classic.append(lin_alg.eigvalsh(in_matrix)[0])
        quantum.append(vqe_result)
        tot_times.append(tot_time)

    MSE = np.square(np.subtract(classic,quantum)).mean()
    RMSE = math.sqrt(MSE)
    NRMSE = RMSE/(max(classic)-min(classic))*100
    print("RMSE: "+str(RMSE))
    print("NRMSE: "+str(NRMSE))


    return [q_shots, opt_iter, NRMSE, mean(tot_times)]

def execute_mde_eigenvalue(ansatz, matrix_op, backend, noise_model=None):
    print(ansatz.__class__.__name__)
    trans_dict = {}
    if not backend.configuration().simulator:
        trans_dict = {'layout_method': 'sabre', 'routing_method': 'sabre'}
    trans_circ = transpile(ansatz, backend, optimization_level=3, **trans_dict)
    vqe_inputs = {
        'ansatz': trans_circ,
        'operator': MatrixOp(matrix_op),
        'optimizer': GradientDescent(maxiter=1000),
        #'noise_model':noise_model,
        #'initial_point': np.random.random(ansatz.num_parameters),
        'initial_point': np.random.random(ansatz.num_parameters)*-1,
        'shots': 8192,
        'measurement_error_mitigation': True
    }
    options = {
        'backend_name': backend.name(),
    }

    job = provider.runtime.run(program_id='vqe', inputs=vqe_inputs, options=options)
    while job.status() != JobStatus.RUNNING and job.status() != JobStatus.ERROR:
        pass
    start = time.time()
    try:
        res = job.result()
        end = time.time()
    except RuntimeJobFailureError as ex:
        print("Job failed!: {}".format(ex))
        print(job.stream_results())
    return [-np.real(res['eigenvalue']), end-start]

def generate_random_matrices(rows, cols, seed, number):
    matrix_list = []
    np.random.seed(seed)
    for i in range(number):
        rdist = np.random.rand(rows, cols)
        rdistt = np.transpose(rdist)
        z1 = np.zeros((rows, cols))
        z2 = np.zeros((rows, cols))
        matrix = np.block([[z1, rdist], [rdistt, z2]])
        matrix_list.append(matrix)

    return matrix_list

def generate_ansaetze(num_qubits, num_reps, entanglement):
    ansaetze = []
    #ansaetze.append(RealAmplitudes(num_qubits, reps=num_reps, entanglement=entanglement))
    ansaetze.append(EfficientSU2(num_qubits, reps=num_reps, entanglement=entanglement))
    #ansaetze.append(TwoLocal(num_qubits))
    #ansaetze.append(PauliTwoDesign(num_qubits, reps=num_reps))
    #ansaetze.append(ExcitationPreserving(num_qubits, reps=num_reps, entanglement=entanglement))
    return ansaetze


target_backend = []
target_backend.append('ibmq_qasm_simulator')
#target_backend.append('ibmq_manila')
#target_backend.append('ibmq_santiago')
#target_backend.append('ibm_perth')
#target_backend.append('ibm_lagos')
#target_backend.append('ibm_jakarta')



# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    IBMQ.load_account()
    provider = IBMQ.get_provider('ibm-q-research-2', 'vienna-uni-tech-1', 'main')

    exponents = range(0,5)
    reps = range(9,11)

    for num_reps in reps:
        for e in exponents:
            num_qubits = e+1
            matrices = generate_random_matrices(2**e, 2**e, 0, 10)
            ansaetze = generate_ansaetze(num_qubits, num_reps, 'linear')

            row = []
            for a in ansaetze:
                row.append(a.__class__.__name__+'-RT')
                row.append(a.__class__.__name__+'-EIG')
            row.append("Classic-EIG")

            for backend_name in target_backend:
                print(backend_name)
                vqe_times = []
                vqe_res = []
                classic_result = []
                summary_fname = "VQE_" + backend_name + '_' + str(num_qubits) + '_' + str(num_reps) + '_rt-nrmse.csv'
                rawdata_fname = "VQE_" + backend_name + '_' + str(num_qubits) + '_' + str(num_reps) + '_raw-data.csv'

                summary = open(summary_fname, 'w')
                rawdata = open(rawdata_fname, 'w')
                summary_writer = csv.writer(summary)
                rawdata_writer = csv.writer(rawdata)

                results = {}
                results = dict.fromkeys(row)
                for r in row:
                    results[r] = []

                summary_writer.writerow(["PQC", "AVG-RUNTIME", "MAE", "RMSE", "NRMSE"])
                rawdata_writer.writerow(row)

                backend = provider.get_backend(backend_name)
                #automatize benchmarks
                for matrix in matrices:
                    # matrix_op = pauli_representation(matrix)
                    results["Classic-EIG"].append(lin_alg.eigvalsh(matrix)[0])
                    for ansatz in ansaetze:
                        key_rt = ansatz.__class__.__name__+"-RT"
                        key_eig = ansatz.__class__.__name__+"-EIG"
                        mde_eig_result = execute_mde_eigenvalue(ansatz, matrix, backend)
                        eig = mde_eig_result[0]
                        rt = mde_eig_result[1]
                        results[key_eig].append(eig)
                        results[key_rt].append(rt)

                    row_results = []
                    for r in row:
                        row_results.append(str(results[r][-1]))

                    rawdata_writer.writerow(row_results)


                print(backend.name())
                print("PQC \t RUNTIME \t MAE \t RMSE \t NRMSE")
                for a in ansaetze:
                    rt = a.__class__.__name__ + '-RT'
                    avg_time = sum(results[rt]) / len(results[rt])
                    eig = a.__class__.__name__ + '-EIG'
                    eig_mae = mean_absolute_error(results["Classic-EIG"], results[eig])
                    eig_rmse = math.sqrt(mean_squared_error(results["Classic-EIG"], results[eig]))
                    eig_nrmse = eig_rmse / (max(results["Classic-EIG"]) - min(results["Classic-EIG"]))
                    print(a.__class__.__name__+" \t" + str(avg_time) + "\t" +
                          str(eig_mae) + "\t" + str(eig_rmse) + "\t" + str(eig_nrmse))

                    summary_writer.writerow([a.__class__.__name__, str(avg_time), str(eig_mae), str(eig_rmse), str(eig_nrmse)])

                summary.close()
                rawdata.close()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
