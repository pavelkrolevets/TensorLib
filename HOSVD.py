
# Author: Pavel Krolevets @ Shanghai Jiao Tong University. e-mail: pavelkrolevets@sjtu.edu.cn
# License: 3-clause BSD.


import numpy as np
from sklearn.utils.extmath import randomized_svd


def unfold(X, mode):
    """This is a tool function to unfold a 3d tensor. Doesnt acept tensors of higher dimentions
    mode = 0 - Mode of unfolding the tensor. Can have values 0,1,2 - as an input tensor has 3 dimensions.
    This approach deffers from the matematical theory, where modes start from 1.
    """
    x, y, z = X.shape
    print('Shape of an input tensor: ', x, y, z)
    if mode == 0:
        G = np.zeros((y, x*z), dtype=float)
        print(G.shape)
        k=0
        for x_1 in range (0, x):
            for z_1 in range (0, z):
                k = k + 1
                G[:, k-1] = X[x_1, :, z_1]


    if mode == 1:
        G = np.zeros((z, x*y), dtype=float)
        print(G.shape)
        k=0
        for i in range (0, x):
            for j in range (0, y):
                k=k+1
                G[:, k-1] = X[i, j, :]


    if mode == 2:
        G = np.zeros((x, y*z), dtype=float)
        print(G.shape)
        k=0
        for i in range (0, z):
            for j in range (0, y):
                k=k+1
                G[:, k-1] = X[:, j, i]

    #print('Unfolded tensor - matrix dimentions: ', G.shape, '\n', 'Matrix: ', G)
    return G


def fold(G ,deph, rows, columns, mode):
    """This is a tool function to fold a 3d tensor from an unfolded tensor.
    i.e. fold(unfold(X, mode=0) = X.
    mode - should be equal unfold mode.
    """
    # deph, rows, columns = 2, 3, 4
    rows_mat, columns_mat = G.shape
    X = np.zeros((deph, rows, columns), float)

    print('Shape of an input matrix: ', rows_mat, columns_mat)
    if mode == 0:
        # G = np.zeros((y, x*z), dtype=float)
        print(G.shape)
        k=0
        for x_1 in range (0, deph):
            for z_1 in range (0, columns):
                k = k + 1
                X[x_1, :, z_1] = G[:, k-1]

    if mode == 1:
        k=0
        for i in range (0, deph):
            for j in range (0, rows):
                k=k+1
                X[i, j, :] = G[:, k-1]

    if mode == 2:

        k=0
        for i in range (0, columns):
            for j in range (0, rows):
                k=k+1
                X[:, j, i] = G[:, k-1]

    #print('Output tensor shape: ', X.shape)
    return X

def HOSVD (X):

    """
    Computes then full HOSVD of a tensor in the form A = S*U1*U2*U3
    (in other words S mode-1 product U1, S mode-2 product U3, S mode-3 product U3)

    Input: any real 3D tensor

    Output: S - 3D tensor with singular values, U1, U2, U3 - corresponding singular vectors

    """

    X1 = unfold(X, mode=0)
    X2 = unfold(X, mode=1)
    X3 = unfold(X, mode=2)

    U1, _, _ = randomized_svd(X1, n_components=X1.shape[0], n_oversamples=10, n_iter='auto',
                              power_iteration_normalizer='auto', transpose='auto',
                              flip_sign=True, random_state=42)
    U2, _, _ = randomized_svd(X2, n_components=X2.shape[0], n_oversamples=10, n_iter='auto',
                              power_iteration_normalizer='auto', transpose='auto',
                              flip_sign=True, random_state=42)
    U3, _, _ = randomized_svd(X3, n_components=X3.shape[0], n_oversamples=10, n_iter='auto',
                              power_iteration_normalizer='auto', transpose='auto',
                              flip_sign=True, random_state=42)

    S = np.dot(np.transpose(U2), unfold(X,1))
    S = fold(S, X.shape[0], X.shape[1], X.shape[2], 1)
    S = np.dot(np.transpose(U1), unfold(S, 0))
    S = fold(S, X.shape[0], X.shape[1], X.shape[2], 0)
    S = np.dot(np.transpose(U3), unfold(S,2))
    S = fold(S, X.shape[0], X.shape[1], X.shape[2], 2)

    return S, U1, U2, U3

def tensor_from_SVD(S, U1, U2, U3, X):
    """
    Computes tensor back from its HOSVD.

    """
    x , y , z = X.shape
    A = np.dot(U1, unfold(S,0))
    A = fold(A, x , y, z, 0)
    A = np.dot(U2, unfold(A, 1))
    A = fold(A, x, y, z, 1)
    A = np.dot(np.transpose(U3), unfold(A, 2))
    A = fold(A, x, y, z, 2)

    return A


