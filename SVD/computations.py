import numpy as np
from numpy import linalg as LA
import scipy
from scipy.sparse import load_npz

def energy(sigma, reqd_energy):
    """
    Calculates the number of singular values to retain in order to ensure the specified energy is retained.
    :param sigma: singular values array
    :param reqd_energy: energy to retain
    :return: num_singular_vals
    """
    # if sigma.ndim == 2:
    #     sigma = np.squeeze(sigma)
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
    :return: U.sigma.Vtr | the reconstructed matrix A after svd
    """
    # get eigen values and vectors
    eig_vals, eig_vecs = LA.eig( np.dot(A.T, A))
    eig_vals = np.absolute(np.real(eig_vals))
    print('eigen values: {}\n\neigen vectors: {}'.format(eig_vals, eig_vecs))

    # calculate the number of eigen values to retain
    if energy_retain == 100:
        eig_vals_num = LA.matrix_rank( np.dot(A.T, A))
    else:
        # sort eigen values in increasing order and compute the number of eigen values to be retained
        eig_vals_num = energy(np.sort(eig_vals)[::-1], energy_retain)

    print('No of eigenvalues retained:{}'.format(eig_vals_num))

    # place the eigen vectors according to increasing order of their corresponding eigen values to form V
    eig_vecs_num = np.argsort(eig_vals)[::-1][0:eig_vals_num]  # TODO
    V = np.real(eig_vecs[:, eig_vecs_num])

    # Calculation of sigma | sort in decreasing order and fill till number of eigen values to retain
    sigma_vals = np.reshape(np.sqrt(np.sort(eig_vals)[::-1])[0:eig_vals_num], eig_vals_num) # removed deepcopy
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


if __name__ == '__main__':
    ex_mat = [[1, 1, 1, 0, 0],[3,3,3,0,0],[4,4,4,0,0],[5,5,5,0,0],[0,2,0,4,4],[0,0,0,5,5],[0,1,0,2,2]]
    ex_mat = np.array(ex_mat)
    A = scipy.sparse.load_npz('../datasets/data_util/A_100k.npz').todense()
    A = np.array(A)
    print(A)
    u, s, vt = compute_svd(A)
    recons = np.array(np.dot(np.dot(u, s), vt))
    print(RMSE(recons, A))