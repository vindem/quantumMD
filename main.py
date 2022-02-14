import numpy as np
from scipy.spatial import distance
import scipy.linalg as lin_alg
from qiskit.aqua.algorithms import VQE
from qiskit import IBMQ, Aer
from qiskit.aqua import QuantumInstance
from qiskit.providers.aer.noise import NoiseModel
from qiskit.ignis.mitigation.measurement import CompleteMeasFitter
from timeit import default_timer as timer
import mda_trajectory_helper as a4md_traj_help
from qiskit.aqua.components.optimizers import COBYLA, SPSA, SLSQP
import os

an_times = []
an_write_times = []
AA_per_seg=2
n_processes=16
n_atimes=1
QUANTUM = False

### NEEDED BY QUANTUM INITIALIZATION
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q-research-2', group='vienna-uni-tech-1', project='main')
backend = Aer.get_backend("qasm_simulator")
device = provider.get_backend("ibm_perth")
coupling_map = device.configuration().coupling_map
# Extracting noise model for error mitigation
noise_model = NoiseModel.from_backend(device.properties())
quantum_instance = QuantumInstance(backend=backend,
                                   shots=8192,
                                   noise_model=noise_model,
                                   coupling_map=coupling_map,
                                   measurement_error_mitigation_cls=CompleteMeasFitter,
                                   cals_matrix_refresh_period=30)
optimizer = SPSA(maxiter=100)
### END


def extract_qubit_op(block_matrix):
    pass

def generate_var_form(qubit_op):
    pass


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
    an_time = timer() - t
    an_times.append(an_time)

    t = timer()
    # ---------============= analysis output code goes here (start) ===========-------------------

    # ---------============= analysis output code goes here (end)   ===========-------------------
    an_write_time = timer() - t
    an_write_times.append(an_write_time)

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

    qubit_op = extract_qubit_op(bpm)
    variational_form = generate_var_form(qubit_op)
    vqe = VQE(qubit_op, variational_form, optimizer=optimizer)
    ret = vqe.run(quantum_instance)
    vqe_result = np.real(ret['eigenvalue'])
    return vqe_result


if __name__ == "__main__":
    print('starting ')
