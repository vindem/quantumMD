import numpy as np
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import PauliTable
from scipy.spatial import distance
import scipy.linalg as lin_alg
from qiskit.aqua.algorithms import VQE, NumPyEigensolver
from qiskit import IBMQ, Aer
from qiskit.aqua import QuantumInstance
from qiskit.providers.aer.noise import NoiseModel
from qiskit.ignis.mitigation.measurement import CompleteMeasFitter
from timeit import default_timer as timer
from statistics import mean
import time
#import mda_trajectory_helper as a4md_traj_help
from qiskit.aqua.components.optimizers import COBYLA, SPSA
from qiskit.aqua.operators import MatrixOperator, WeightedPauliOperator, op_converter
from qiskit.opflow.primitive_ops import MatrixOp
import math
import csv
from datetime import datetime
from qiskit.chemistry import FermionicOperator
from qiskit.opflow import Z, I
from qiskit.providers import JobStatus
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.opflow.state_fns import OperatorStateFn
### NEEDED BY QUANTUM INITIALIZATION



### END


n_atimes = 100
QUANTUM = False
largest_eig_vals_by_frame = []
def analyze(types, xpoints, ypoints, zpoints, box_points, step, optimizer = None, quantum_instance = None):
    print('-----======= Python : analyze ({})========-------'.format(step))
    print(f"Usable cpus = {os.sched_getaffinity(0)}")

    global atom_index_groups
    t = timer()
    # ---------============= analysis code goes here (start) ===========-------------------
    points = np.vstack((xpoints, ypoints, zpoints)).T
    print(points.shape)
    if atom_index_groups is None:
        top = a4md_traj_help.get_topology("./reference.pdb")
        # atom_groups = a4md_traj_help.get_atoms_groups(top, group_method='alpha_carbon')
        atom_groups = a4md_traj_help.get_atoms_groups(top, group_method='backbone')
        print('Number of carbon alpha atoms : {}'.format(len(atom_groups)))
        atom_index_groups = a4md_traj_help.get_segments(atom_groups, segment_length=AA_per_seg, include_last=True)
        print('Number of carbon alpha atoms groups : {}'.format(len(atom_index_groups)))

    levs = []
    segs = []
    nsegs = len(atom_index_groups)
    for i in range(nsegs):
        for j in range(i + 1, nsegs):
            seg1 = points[atom_index_groups[i]]
            seg2 = points[atom_index_groups[j]]
            segs.append([seg1, seg2])

    #with mp.Pool(processes=n_processes) as pool:
      #  for i in range(n_atimes):
       #     result = pool.map_async(calc_eigval, segs)
        #    levs = result.get()

    for i in range(n_atimes):
        if QUANTUM:
            levs = calc_eigval_quantum(segs, optimizer, quantum_instance)
        else:
            levs = calc_eigval_classic(segs)

    largest_eig_vals_by_frame.append(levs)
    # ---------============= analysis code goes here (end)   ===========-------------------
    #an_time = timer() - t
    #an_times.append(an_time)

    t = timer()
    # ---------============= analysis output code goes here (start) ===========-------------------

    # ---------============= analysis output code goes here (end)   ===========-------------------
    #an_write_time = timer() - t
    #an_write_times.append(an_write_time)

    # if step+1>=nsteps:
    #    print('------============ reached end of analysis ({}) ==========------------'.format(step))
    #    with open('largest_eig_values_bmatrix_by_frame.pickle', 'wb') as f:
    #        pickle.dump(largest_eig_vals_by_frame, f)

    #    job_document = {}
    #    job_document['step_analysis_time_s'] = " ".join(str(at) for at in an_times)
    #    job_document['total_analysis_time_s'] = np.sum(an_times)
    #    job_document['sem_analysis_time_s'] = stats.sem(an_times)
    #    job_document['total_analysis_output_time_s'] = np.sum(an_write_times)
    #    job_document['sem_analysis_output_time_s'] = stats.sem(an_write_times)

    #    with open('job_document.json', 'w') as f:
    #        f.write(json.dumps(job_document))

    print('-----======= Python : analyze done for ({}, time: {} s)========-------'.format(step, an_time))

    return 0

def calc_eigval_classic(segs):
    #print(f"{_location()}, affinity before thread_foo:"
    #      f" {os.sched_getaffinity(0)}")
    seg1 = segs[0]
    seg2 = segs[1]
    seg1_l = len(seg1)
    seg2_l = len(seg2)
    d = distance.cdist(seg1, seg2, 'sqeuclidean')
    dt = np.transpose(d)
    z1 = np.zeros((seg1_l,seg1_l))
    z2 = np.zeros((seg2_l,seg2_l))
    bpm = np.block([[z1,d],[dt,z2]])
    lev = lin_alg.eigvalsh(bpm)[-1]
    return lev

def calc_eigval_quantum(segs, optimizer, quantum_instance):
    seg1 = segs[0]
    seg2 = segs[1]
    seg1_l = len(seg1)
    seg2_l = len(seg2)
    d = distance.cdist(seg1, seg2, 'sqeuclidean')
    dt = np.transpose(d)
    z1 = np.zeros((seg1_l, seg1_l))
    z2 = np.zeros((seg2_l, seg2_l))
    bpm = np.block([[z1, d], [dt, z2]])

    qubit_op = None
    variational_form = None
    vqe = VQE(qubit_op, variational_form, optimizer=optimizer)
    ret = vqe.run(quantum_instance)
    vqe_result = np.real(ret['eigenvalue'])
    return vqe_result






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

if __name__ == "__main__":
    print('starting ')
    outfile = open('runtime_tests_real.csv','w')
    writer = csv.writer(outfile)
    opt_iters = [50,100,500,1000]
    writer.writerow(['shots', 'opt_iter', 'nrmse', 'tot_time'])
    for it in opt_iters:
        ret = test_quantum_eigenvalue(opt_iter=it)
        writer.writerow(ret)

    outfile.close()
