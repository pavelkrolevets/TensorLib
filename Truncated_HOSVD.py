
""" Author: Pavel Krolevets @ Shanghai Jiao Tong University
    e-mail: pavelkrolevets@sjtu.edu.cn """


import numpy as np
from sklearn.utils.extmath import randomized_svd


A = [[0.9073, 0.7158,-0.3698, 1.7842, 1.6970, 0.0151, 2.1236,-0.0740, 1.4429],
     [0.8924,-0.4898, 2.4288, 1.7753,-1.5077, 4.0337,-0.6631, 1.9103,-1.7495],
     [2.1488, 0.3054, 2.3753, 4.2495, 0.3207, 4.7146, 1.8260, 2.1335,-0.2716]]

B = np.zeros((2,3,4))
B[0,:,:] = [[1,4,7,10],
            [2,5,8,11],
            [3,6,9,12]]
B[1,:,:] = [[13,16,19,22],
            [14,17,20,23],
            [15,18,21,24]]


def unfold(X, mode):
    """This is a tool function to unfold a 3d tensor. Doesnt acept tensors of higher dimentions
    mode = 0 - Mode of unfolding the tensor. Can have values 0,1,2 - as an input tensor has 3 dimensions.
    This approach deffers from the matematical theory, where modes start from 1.
    """
    z, x, y = X.shape

    if mode == 0:
        G = np.zeros((x,0), dtype=float)
        for i in range(z):
            G = np.concatenate((G, X[i,:,:]), axis=1)

    if mode == 1:
        G = np.zeros((y,0), dtype=float)
        for i in range(z):
            G = np.concatenate((G, X[i,:,:].T), axis=1)

    if mode == 2:
        G = np.zeros((z,0), dtype=float)
        for i in range(y):
            G = np.concatenate((G, X[:,:,i]), axis=1)
    return  G



def fold(G ,z, x, y, mode):
    """This is a tool function to fold a 3d tensor from an unfolded tensor.
    i.e. fold(unfold(X, mode=0) = X.
    mode - should be equal unfold mode.
    """
    # deph, rows, columns = 2, 3, 4
    rows_mat, columns_mat = G.shape
    X = np.zeros((z, x, y), float)

    print('Shape of an input matrix: ', rows_mat, columns_mat)
    if mode == 0:
        for i in range(z):
            X[i,:,:] = G[:,i*y : y+i*y]
    if mode == 1:
        for i in range(z):
            X[i,:,:] = G[:,i*x: x+i*x].T
    if mode == 2:
        for i in range(y):
            X[:,:,i] = G[:,i*x: x+i*x]

    return X

A = np.asarray(A)
print(A.shape)
A = fold(A,3,3,3,0)


X1 = unfold(A, mode=0)
X2 = unfold(A, mode=1)
X3 = unfold(A, mode=2)

U1, S1, V1 = randomized_svd(X1, n_components=3, n_oversamples=10, n_iter='auto',
                   power_iteration_normalizer='auto', transpose='auto',
                   flip_sign=True, random_state=42)
U2, S2, V2 = randomized_svd(X2, n_components=3, n_oversamples=10, n_iter='auto',
                   power_iteration_normalizer='auto', transpose='auto',
                   flip_sign=True, random_state=42)
U3, S3, V3 = randomized_svd(X3, n_components=3, n_oversamples=10, n_iter='auto',
                   power_iteration_normalizer='auto', transpose='auto',
                   flip_sign=True, random_state=42)


S1 = np.dot(U1.T, X1)
S1 = fold(S1, A.shape[0], A.shape[1], A.shape[2], 0)
S2 = np.dot(np.transpose(U2), unfold(S1, 1))
S2 = fold(S2, A.shape[0], A.shape[1], A.shape[2], 1)
S3 = np.dot(np.transpose(U3), unfold(S2, 2))
S3 = fold(S3, A.shape[0], A.shape[1], A.shape[2], 2)

np.set_printoptions(3)
print(unfold(S3,0))

x , y , z = S3.shape
Z1 = np.dot(U1, unfold(S3, 0))
Z1 = fold(Z1, x , y, z, 0)
Z2 = np.dot(U2, unfold(Z1, 1))
Z2 = fold(Z2, x, y, z, 1)
Z3 = np.dot(U3, unfold(Z2, 2))
Z3 = fold(Z3, x, y, z, 2)

np.set_printoptions(precision=4, suppress=True)
norm_x = np.square(np.linalg.norm(unfold(A,0), ord='fro'))
print(norm_x, np.square(np.linalg.norm(unfold(Z3,0), ord='fro')),'\n', unfold(A, 0), '\n', unfold(Z3, 0))

