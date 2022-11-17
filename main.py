import os

import numpy as np
from qiskit.circuit.library import RealAmplitudes
import os

from scipy.spatial import distance
import scipy.linalg as lin_alg

from quantum_api import calc_eigval_quantum, calculate_distance_quantum

from qiskit import IBMQ, Aer
from qiskit.providers.aer.noise import NoiseModel
from qiskit.ignis.mitigation.measurement import CompleteMeasFitter
from timeit import default_timer as timer
from statistics import mean
import time
import mda_trajectory_helper as a4md_traj_help

import math
import csv

### END

def extract_bpm(segs, dist_function):
    seg1 = segs[0]
    seg2 = segs[1]
    seg1_l = len(seg1)
    seg2_l = len(seg2)
    d = dist_function(seg1, seg2)
    dt = np.transpose(d)
    z1 = np.zeros((seg1_l, seg1_l))
    z2 = np.zeros((seg2_l, seg2_l))
    bpm = np.block([[z1, d], [dt, z2]])

def classic_euclidean_distance(A,B):
    distance.cdist(A, B, 'sqeuclidean')

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
    #    for i in range(n_atimes):
    #        result = pool.map_async(calc_eigval, segs)
    #        levs = result.get()

    for i in range(n_atimes):
        if QUANTUM:
            bpm = extract_bpm(segs, calculate_distance_quantum)
            levs = calc_eigval_quantum(segs, optimizer, quantum_instance)
        else:
            bpm = extract_bpm(segs, classic_euclidean_distance)
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
