import numpy as np
from numpy import linalg as LA
import scipy
from scipy.sparse import load_npz
from collections import Counter
import math
from copy import deepcopy

def energy(sigma, reqd_energy):
    """
    Calculates the number of singular values to retain in order to ensure the specified energy is retained.
    :param sigma: singular values array
    :param reqd_energy: energy to retain
    :return: num_singular_vals
    """
    if sigma.ndim == 2:
        sigma = np.squeeze(sigma)
    if reqd_energy == 0:
        return -1
    # calculate total sum
    total_energy = np.sum(sigma)

    # calculate percent energy
    percentage_reqd_energy = reqd_energy * total_energy / 100.0

    # calculate cumulative sum of the singular values in non-decreasing order
    indexSum = sigma.cumsum()
    checkSum = indexSum <= percentage_reqd_energy

    # number of singular values to consider
    last_singular_val_pos = np.argmin(checkSum)
    num_singular_vals = last_singular_val_pos + 1
    return num_singular_vals


def compute_svd(A, energy_retain=90):
    """

    :param A: matrix to be decomposed
    :param energy_retain: energy to retain
    :return: U, sigma, Vtr
    """
    # get eigen values and vectors
    eig_vals, eig_vecs = LA.eig( np.dot(A.T, A))
    eig_vals = np.absolute(np.real(eig_vals))
    print('eigen values: {}\n\neigen vectors: {}'.format(eig_vals, eig_vecs))

    # calculate the number of eigen values to retain
    if energy_retain == 100:
        eig_vals_num = LA.matrix_rank(np.dot(A.T, A))
    else:
        # sort eigen values in increasing order and compute the number of eigen values to be retained
        eig_vals_num = energy(np.sort(eig_vals)[::-1], energy_retain)

    print('No of eigenvalues retained:{}'.format(eig_vals_num))

    # place the eigen vectors according to increasing order of their corresponding eigen values to form V
    eig_vecs_num = np.argsort(eig_vals)[::-1][0:eig_vals_num]  # TODO
    V = np.real(eig_vecs[:, eig_vecs_num])

    # Calculation of sigma | sort in decreasing order and fill till number of eigen values to retain
    sigma_vals = deepcopy(np.reshape(np.sqrt(np.sort(eig_vals)[::-1])[0:eig_vals_num], eig_vals_num)) # removed deepcopy
    sigma = np.zeros([eig_vals_num, eig_vals_num])
    np.fill_diagonal(sigma, sigma_vals)

    # Calculation of U by using U = AVS^-1
    U = np.dot(A, np.dot(V, LA.inv(sigma)))

    Vtr = V.T
    print("U: {}".format(U))
    print("sigma: {}".format(sigma))
    print("V_transpose: {}".format(Vtr))
    return U, sigma, Vtr


def RMSE(matA, matB):
    """
    Calculates the rmse between two matrices
    :param matA:
    :param matB:
    :return: root mean square error value
    """
    return np.sqrt(np.mean(np.square(matA - matB)))  # need to check whether same ans with pred, truth


def spearman(matA, matB):
    """
    Calculates the spearman rank correlation of two matrices
    :param matA:
    :param matB:
    :return: spearman rank correlation value
    """
    d = np.sum(np.square(matA - matB))
    result = 1-6.0*d/(len(matA)*(len(matA)*len(matA)-1))
    return result


def row_col_selection(A, r, repeat):
    """
    Function for row/column selection from A to form R/C respectively
    :param A: matrix to be decomposed
    :param r: number of selections
    :param repeat: is repetitive selection allowed or not
    :return: selected rows, R
    """
    index_set = [i for i in range(len(A))]
    frob = 0

    # compute frobenius norm for A
    for i in range(len(A)):
        for j in range(len(A[i])):
            frob += A[i][j] ** 2

    # compute prob for random selection of rows and columns
    prob = np.zeros(len(A))
    for i in range(len(A)):
        sum_sqr_row_vals = 0
        for j in range(len(A[i])):
            sum_sqr_row_vals += A[i][j]**2
        prob[i] = sum_sqr_row_vals / float(frob)

    sel_rows = np.random.choice(index_set, r, repeat, prob)

    # form R/C with random selected rows/columns
    R = np.zeros((r, len(A[0])))
    for i, row in zip(range(r), sel_rows):
        for j in range(len(A[row])):
            R[i][j] = A[row][j]
            R[i][j] = R[i][j]/float(math.sqrt(r*prob[row]))

    return sel_rows, R


def compute_U(A, r, row_idx, col_idx):
    """
    Computation of U using Moore-Pennrose pseudoinverse
    :param A: matrix to be decomposed
    :param r: number of row/col selection
    :param row_idx: set of selected row indices
    :param col_idx: set of selected column indices
    :return: U
    """
    # Form W by intersection of C and R
    W = np.zeros((r, r))
    for i, row in zip(range(len(row_idx)), row_idx):
        for j, column in zip(range(len(col_idx)), col_idx):
            W[i][j] = A[row][column]

    # Compute pseudo-inverse of W
    X, eig_vals, Ytr = LA.svd(W, full_matrices=True)

    sig = np.zeros((r, r))
    sig_plus = np.zeros((r, r))

    # replace non-zero sigma values with its inverse
    for i in range(len(eig_vals)):
        sig[i][i] = math.sqrt(eig_vals[i])
        if sig[i][i] != 0:
            sig_plus[i][i] = 1/float(sig[i][i])

    # finally compute U with the given formula
    U = np.dot(np.dot(Ytr.T, np.dot(sig_plus, sig_plus)), X.T)

    return U


def compute_cur(A, r):
    """
    Main function to comput C, U, R
    :param A: matrix to be decomposed
    :param r: number of row/col selection
    :return: C, U, R
    """
    row_idx, tmpR = row_col_selection(A, r, False)
    col_idx, tmpC = row_col_selection(A.T, r, False)
    R = tmpR
    C = tmpC.T

    # print(R)
    U = compute_U(A, r, row_idx, col_idx)
    print(np.dot(np.dot(C, U), R))

    return C, U, R


if __name__ == '__main__':
    ex_mat = [[1,1,1,0,0],[3,3,3,0,0],[4,4,4,0,0],[5,5,5,0,0],[0,0,0,4,4],[0,0,0,5,5],[0,0,0,2,2]]
    ex_mat = np.array(ex_mat)
    A = scipy.sparse.load_npz('../datasets/data_util/A_100k.npz').todense()
    A = np.array(A)
    compute_cur(ex_mat, 3)
    # print(A)
    # u, s, vt = compute_svd(A)
    # recons = np.array(np.dot(np.dot(u, s), vt))
    # print(RMSE(recons, A))