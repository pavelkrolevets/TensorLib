
# Author: Pavel Krolevets @ Shanghai Jiao Tong University. e-mail: pavelkrolevets@sjtu.edu.cn
# License: 3-clause BSD.


import numpy as np
from sklearn.utils.extmath import randomized_svd
from HOSVD import fold, unfold, HOSVD, tensor_from_SVD

A = [[0.9073, 0.7158,-0.3698, 1.7842, 1.6970, 0.0151, 2.1236,-0.0740, 1.4429],
     [0.8924,-0.4898, 2.4288, 1.7753,-1.5077, 4.0337,-0.6631, 1.9103,-1.7495],
     [2.1488, 0.3054, 2.3753, 4.2495, 0.3207, 4.7146, 1.8260, 2.1335,-0.2716]]

A = np.asarray(A)
print(A.shape)
A = fold(A,3,3,3,0)

X = A

X1 = unfold(X, mode=0)
X2 = unfold(X, mode=1)
X3 = unfold(X, mode=2)

U1, S1, V1 = randomized_svd(X1, n_components=3, n_oversamples=10, n_iter='auto',
                   power_iteration_normalizer='auto', transpose='auto',
                   flip_sign=True, random_state=42)
U2, S2, V2 = randomized_svd(X2, n_components=3, n_oversamples=10, n_iter='auto',
                   power_iteration_normalizer='auto', transpose='auto',
                   flip_sign=True, random_state=42)
U3, S3, V3 = randomized_svd(X3, n_components=3, n_oversamples=10, n_iter='auto',
                   power_iteration_normalizer='auto', transpose='auto',
                   flip_sign=True, random_state=42)


S = np.dot(np.transpose(U1), unfold(X, 0))
S = fold(S, X.shape[0], X.shape[1], X.shape[2], 0)
S = np.dot(np.transpose(U2), unfold(S, 1))
S = fold(S, X.shape[0], X.shape[1], X.shape[2], 1)
S = np.dot(np.transpose(U3), unfold(S, 2))
S = fold(S, X.shape[0], X.shape[1], X.shape[2], 2)

print(U1)

x , y , z = S.shape
Z = np.dot(U3, unfold(S, 2))
Z = fold(Z, x , y, z, 2)
Z = np.dot(U2, unfold(S, 1))
Z = fold(Z, x, y, z, 1)
Z = np.dot(U1, unfold(S, 0))
Z = fold(Z, x, y, z, 0)

np.set_printoptions(precision=4, suppress=True)
norm_x = np.square(np.linalg.norm(unfold(A,0), ord='fro'))
print(norm_x, np.square(np.linalg.norm(unfold(Z,0), ord='fro')),'\n', unfold(A, 0), '\n', unfold(Z, 0))

