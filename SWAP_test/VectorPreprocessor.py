import numpy as np
import time
from scipy.spatial import distance


class VectorPreprocessor:
    # Defining the constructor for our MD problem. The variables of the constructor are Segments A and  B

    def __init__(self, A, B):
        self.A = A
        self.B = B

        # Preprocessing part - If the 1d array length is 2, don't pad 0, if length is 3, pad 0 to make it 2**2 entry vector

    def vec(self):
        arr1 = np.array(self.A)
        arr2 = np.array(self.B)
        zeros = np.zeros(len(arr1))
        rsf = arr1.shape[1]

        if arr1.shape == (2,) and arr2.shape == (2,):
            return arr1, arr2

        if arr1.shape[1] == 2 and arr2.shape[1] == 2:
            return arr1, arr2

        # Padding 0's to the columns to make it 2**n in dimension. For our case 2**2 = 4 should do it for the
        # amplitude encoding. This forms a part of the preprocessing
        else:

            A1 = np.column_stack((arr1, zeros))
            A2 = np.column_stack((arr2, zeros))
        return A1, A2, rsf

    def classic_eucl_dist(self):
        st = time.time()
        # computing Euclidean distance between each atom, constructed as a len(A1) x 4 tensor
        dist = [distance.euclidean(self.A[i], self.B[i]) for i in range(len(self.A))]
        return [dist, time.time() - st]

    # Quantum preprocessing part

    # Computing the norm, so that it can be used to normalize the vector
    def norm(self):
        vecA_norm = [np.linalg.norm(self.vec()[0][i]) for i in range(len(self.vec()[0]))]
        vecB_norm = [np.linalg.norm(self.vec()[1][i]) for i in range(len(self.vec()[0]))]
        return [vecA_norm, vecB_norm]

    # The norm factor, ideally is used to normalize the quantum register with 3 qubits.
    def norm_factor(self):
        W_mat = [np.sqrt(self.norm()[0][j] ** 2 + self.norm()[1][j] ** 2) for j in range(len(self.norm()[0]))]
        return W_mat

        # First QuantumRegister state being Amplitude encoded

    def phi_reg(self):
        phi = [np.array([self.norm()[0][i] / (self.norm_factor()[i]), - self.norm()[1][i] / (self.norm_factor()[i])])
               for i in range(len(self.norm()[0]))]
        return phi

    # Second big (3 qubit) normalized quantum register being initialized and amplitude encoded

    def psi_reg(self):
        psi = []
        # psi = np.empty([len(self.vec()[0]), len(self.vec()[0]) + len(self.vec()[1])])
        for i in range(len(self.vec()[0])):
            psi.append((self.vec()[0][i]) / (self.norm()[0][i] * np.sqrt(2)))
            psi.append((self.vec()[1][i]) / (self.norm()[1][i] * np.sqrt(2)))
        #print(" ")
        return np.array(psi).reshape(len(self.A), 2 ** (self.vec()[2]))

# This completes with the prepocessing part of the CSWAP test from a classical perspective