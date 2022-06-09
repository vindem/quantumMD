import timeit

from qiskit.circuit.library import EfficientSU2
from qiskit import IBMQ
from qiskit.algorithms.optimizers import GradientDescent, COBYLA
import numpy as np
from qiskit.opflow import MatrixOp
from qiskit.algorithms import VQE
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.test.mock import FakeJakarta, FakeManila, FakeLagos, FakeSantiago
from qiskit import Aer
from qiskit.utils import QuantumInstance
from matplotlib import pyplot as plt
import scipy.linalg as lin_alg
from qiskit.utils.mitigation import CompleteMeasFitter
import csv
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

from main import generate_random_matrices, generate_ansaetze


IBMQ.load_account()
provider = IBMQ.get_provider('ibm-q-research-2', 'vienna-uni-tech-1', 'main')

exponents = range(0,5)
reps = range(1,6)

#target_backend = [FakeJakarta(), FakeManila(), FakeSantiago(), FakeLagos()]
target_backend = [FakeJakarta()]
backend_sim = Aer.get_backend('aer_simulator')

for num_reps in reps:
    for e in exponents:
        num_qubits = e + 1
        matrices = generate_random_matrices(2 ** e, 2 ** e, 0, 10)
        ansaetze = generate_ansaetze(num_qubits, num_reps, 'circular')

        row = []
        for a in ansaetze:
            row.append(a.__class__.__name__ + '-RT')
            row.append(a.__class__.__name__ + '-EIG')
        row.append("Classic-EIG")

        for backend in target_backend:
            print(backend.name())
            vqe_times = []
            vqe_res = []
            classic_result = []
            summary_fname = "VQE_" + backend.name() + '_' + str(num_qubits) + '_' + str(num_reps) + '_rt-nrmse.csv'
            rawdata_fname = "VQE_" + backend.name() + '_' + str(num_qubits) + '_' + str(num_reps) + '_raw-data.csv'

            summary = open(summary_fname, 'w')
            rawdata = open(rawdata_fname, 'w')
            summary_writer = csv.writer(summary)
            rawdata_writer = csv.writer(rawdata)

            results = {}
            results = dict.fromkeys(row)
            for r in row:
                results[r] = []

            summary_writer.writerow(["PQC","AVG-RUNTIME", "MAE", "MSE", "RMSE", "NRMSE"])
            rawdata_writer.writerow(row)



            noise_model = NoiseModel.from_backend(QasmSimulator.from_backend(backend))
            quantum_instance = QuantumInstance(backend=backend_sim, noise_model=noise_model,
                                               measurement_error_mitigation_cls=CompleteMeasFitter)

            optimizer = COBYLA(maxiter=100)


            for matrix in matrices:
                # matrix_op = pauli_representation(matrix)
                operator = MatrixOp(matrix)

                results["Classic-EIG"].append(lin_alg.eigvalsh(matrix)[0])
                for ansatz in ansaetze:
                    initial_point = np.random.random(ansatz.num_parameters)

                    local_vqe = VQE(ansatz=ansatz,
                                    optimizer=optimizer,
                                    initial_point=initial_point,
                                    quantum_instance=quantum_instance)

                    key_rt = ansatz.__class__.__name__ + "-RT"
                    key_eig = ansatz.__class__.__name__ + "-EIG"
                    start = timeit.timeit()
                    mde_eig_result = local_vqe.compute_minimum_eigenvalue(operator)
                    end = timeit.timeit()
                    eig = np.real(mde_eig_result.eigenvalue)
                    rt = end - start
                    results[key_eig].append(eig)
                    results[key_rt].append(rt)

                row_results = []
                for r in row:
                    row_results.append(str(results[r][-1]))

                rawdata_writer.writerow(row_results)

            print(backend.name())
            print("PQC \t RUNTIME \t MAE \t MSE \t RMSE \t NRMSE")
            for a in ansaetze:
                rt = a.__class__.__name__ + '-RT'
                avg_time = sum(results[rt]) / len(results[rt])
                eig = a.__class__.__name__ + '-EIG'
                eig_mae = mean_absolute_error(results["Classic-EIG"], results[eig])
                eig_mse = mean_squared_error(results["Classic-EIG"], results[eig])
                eig_rmse = math.sqrt(mean_squared_error(results["Classic-EIG"], results[eig]))
                eig_nrmse = eig_rmse / (max(results["Classic-EIG"]) - min(results["Classic-EIG"]))
                print(a.__class__.__name__ + " \t" + str(avg_time) + "\t" +
                      str(eig_mae) + "\t" + str(eig_mse) + "\t" + str(eig_rmse) + "\t" + str(eig_nrmse))

                summary_writer.writerow(
                    [a.__class__.__name__, str(avg_time), str(eig_mae), str(eig_mse), str(eig_rmse), str(eig_nrmse)])

            summary.close()
            rawdata.close()
