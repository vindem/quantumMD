import numpy as np
from itertools import product
from qiskit.opflow import MatrixOp, X, Y, Z, I

def pauli_representation(matrix):
    dim = len(matrix)
    next_power_of_two = int(2**np.ceil(np.log(dim)/np.log(2)))
    if next_power_of_two != dim:
        diff = next_power_of_two - dim
        matrix = np.hstack((matrix, np.zeros((dim, diff))))
        matrix = np.vstack((matrix, np.zeros(diff, next_power_of_two)))

    dim = next_power_of_two
    num_tensor_repetitions = int(np.log(dim)/np.log(2))
    pauli_matrices = {
        'I': np.array([[1,0],[0,1]]),
        'X': np.array([[0,1],[1,0]]),
        'Y': np.array([[0,-1j],[1j,0]]),
        'Z': np.array([[1,0],[0,-1]])
    }
    pauli_key_list = []
    keys_to_delete = []
    pauli_dict = {}

    def pauliDictValues(l):
        yield from product(*([l] * num_tensor_repetitions))

    for x in pauliDictValues('IXYZ'):
        pauli_key_list.append(''.join(x))

    for y in pauli_key_list:
        pauli_dict[y] = 0

    for key in pauli_dict:
        temp_list = []
        pauli_tensors = []
        num_ypsilons = key.count('Y')
        temp_key = str(key)

        if num_ypsilons % 2 == 0:
            for string in temp_key:
                temp_list.append(string)

            for spin_matrix in temp_list:
                pauli_tensors.append(pauli_matrices[spin_matrix])
            pauli_dict[key] = pauli_tensors

            current_matrix = pauli_dict[key].copy()

            for k in range(1, num_tensor_repetitions):
                temporary_dict = np.kron(current_matrix[k-1], current_matrix[k])
                current_matrix[k] = temporary_dict

            pauli_dict[key] = current_matrix[-1]

        else:
            keys_to_delete.append(key)

    for val in keys_to_delete:
        pauli_dict.pop(val)

    vec_ham_elements = np.zeros(int((dim**2+dim)/2))
    h = 0
    for i in range(0,dim):
        for j in range(i, dim):
            arr = []
            vec_ham_elements[h] = matrix[i][j]
            for key in pauli_dict:
                temp_var = pauli_dict[key]
                arr.append(float(temp_var[i][j]))

            if i == 0 and j == 0:
                final_mat = np.array(arr.copy())
            else:
                final_mat = np.vstack((final_mat,arr))

            h += 1

    x = np.linalg.solve(final_mat, vec_ham_elements)
    a = []
    var_list = list(pauli_dict.keys())

    operators_map = {
            'I': I,
            'X': X,
            'Y': Y,
            'Z': Z
        }

    pauli_decomposition = 0
    for i in range(len(pauli_dict)):
        b = x[i]
        if abs(b) > 0.00001:
            pauli_op = I
            for c in var_list[i]:
                pauli_op = pauli_op ^ operators_map[c]
            pauli_decomposition += b * pauli_op

    return pauli_decomposition