import os
import sys

import numpy as np
import os

from scipy.spatial import distance
import scipy.linalg as lin_alg
from sklearn.metrics import mean_squared_error

from quantum_api import calc_eigval_quantum, calculate_distance_quantum

from timeit import default_timer as timer
from statistics import mean
import time
import mda_trajectory_helper as a4md_traj_help

import math
import csv
import importlib

#QUANTUM PART
from qiskit import IBMQ, Aer
#from qiskit.providers.aer.noise import NoiseModel
from qiskit.circuit.library import RealAmplitudes
from qiskit.algorithms.optimizers import COBYLA, GradientDescent, SPSA
from qiskit_aqt_provider import AQTProvider
#END



an_times = []
an_write_times = []
n_processes=4
n_atimes=1
global QUANTUM
def extract_bpm(segs, dist_function):
    seg1 = segs[0][0]
    seg2 = segs[1][0]

    seg1_l = len(seg1)
    seg2_l = len(seg2)
    d = dist_function(seg1, seg2)
    #print(d.shape)
    #d = distance.cdist(np.array(seg1), np.array(seg2), 'sqeuclidean')
    #d = distance.cdist(np.array(seg1), np.array(seg2), 'sqeuclidean')
    #d = distance.cdist(segs, 'sqeuclidean')
    dt = np.transpose(d)
    z1 = np.zeros((seg1_l, seg1_l))
    z2 = np.zeros((seg2_l, seg2_l))
    bpm = np.block([[z1, d], [dt, z2]])
    return bpm

def classic_euclidean_distance(A,B):
    arr = np.array(distance.cdist(A, B, 'sqeuclidean'))
    print(arr.shape)
    return arr


largest_eig_vals_by_frame = []
atom_index_groups = None
def analyze(ref_file, traj_file, step, seg_len, QUANTUM=False, ansatz_class=None, backend=None, optimizer=None):
    print('-----======= Python : analyze ({})========-------'.format(step))
    print(f"Usable cpus = {os.sched_getaffinity(0)}")

    global atom_index_groups
    t = timer()
    # ---------============= analysis code goes here (start) ===========-------------------
    points = a4md_traj_help.get_coordinates(ref_file, traj_file, step)
    #print(points.shape)
    if atom_index_groups is None:
        top = a4md_traj_help.get_topology(ref_file)
        # atom_groups = a4md_traj_help.get_atoms_groups(top, group_method='alpha_carbon')
        atom_groups = a4md_traj_help.get_atoms_groups(top, group_method='backbone')
        print('Number of carbon alpha atoms : {}'.format(len(atom_groups)))
        atom_index_groups = a4md_traj_help.get_segments(atom_groups, segment_length=seg_len, include_last=True)
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
    #    for i in range(n_atimes):
    #        result = pool.map_async(calc_eigval, segs)
    #        levs = result.get()

    for i in range(n_atimes):
        if QUANTUM:
            bpm = extract_bpm(segs, calculate_distance_quantum)
            num_qubits = int(math.log2(bpm.shape[0]))
            ansatz = globals()[ansatz_class](num_qubits)
            levs = calc_eigval_quantum(bpm, ansatz, backend, optimizer)
        else:
            bpm = extract_bpm(segs, classic_euclidean_distance)
            levs = lin_alg.eigvalsh(bpm)[-1]

    largest_eig_vals_by_frame.append(levs)
    # ---------============= analysis code goes here (end)   ===========-------------------
    an_time = timer() - t
    an_times.append(an_time)

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
    seg1 = segs[0][0]
    seg2 = segs[1][1]
    seg1_l = len(seg1)
    seg2_l = len(seg2)
    print(np.array(segs).shape)
    d = distance.cdist(seg1, seg2, 'sqeuclidean')
    dt = np.transpose(d)
    z1 = np.zeros((seg1_l,seg1_l))
    z2 = np.zeros((seg2_l,seg2_l))
    bpm = np.block([[z1,d],[dt,z2]])
    lev = lin_alg.eigvalsh(bpm)[-1]
    return lev
# python calc_leigval_bipartite_multithread_offline.py reference.pdb traj_comp.xtc 5 8
if __name__ == "__main__":
    print('--------=========== Running analysis in python ===========-------------')
    ref_file = sys.argv[1]
    traj_file = sys.argv[2]
    step = int(sys.argv[3])
    seg_len = int(sys.argv[4])
    QUANTUM = eval(sys.argv[5])

    q_results = {}
    OPT_ITERS = [20]
    # 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
    # OPT_ITERS = [1, 2]

    if not QUANTUM:
        analyze(ref_file, traj_file, step, seg_len)
    else:
        #OPTIMIZERS = [COBYLA(20), GradientDescent(20), SPSA(20)]
        ansatz = "RealAmplitudes"
        backend_name = "ibmq_qasm_simulator"
        opt_iter = 20
        optimizer = COBYLA(opt_iter)


        key = (optimizer.__class__.__name__, opt_iter)
        q_eigenvalues = []

        q_eig = analyze(ref_file, traj_file, step, seg_len, True, ansatz, backend_name, optimizer)
        q_eigenvalues.append(q_eig)
        q_results[key] = q_eigenvalues

        #https://stackoverflow.com/questions/4821104/dynamic-instantiation-from-string-name-of-a-class-in-dynamically-imported-module

        classic_LEBM = analyze(ref_file, traj_file, step, seg_len)

        print("Classic: "+str(classic_LEBM))
        print("Quantum: "+str(q_eig))

        # postprocessing
        """
        mse_data = {}
        for k in q_results.keys():
            if k[0] not in mse_data:
                mse_data[k[0]] = []
            mse = mean_squared_error([classic_LEBM] * opt_iter, q_results[k])
            mse_data[k[0]].append(mse)

        # writing on file
        
        header = ['Opt-iter']
        for k in mse_data.keys():
            header.append(k)

        with open('outfile.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerow(header)
            j = 0
            for i in OPT_ITERS:
                row = [str(i)]
                for k in mse_data.keys():
                    row.append(str(mse_data[k][j]))
                writer.writerow(row)
                j = j + 1

            file.close()
        """