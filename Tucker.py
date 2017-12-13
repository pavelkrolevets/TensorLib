"""
Author: Pavel Krolevets @ Shanghai Jiao Tong University. e-mail: pavelkrolevets@sjtu.edu.cn
"""


import numpy as np

X = np.arange(1,25).reshape(2, 3, 4)
print(X)
def unfold(X, mode):
    """This is a tool function to unfold a 3d tensor. Dont acept tensors of higher dimentions
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

    print('Unfolded tensor - matrix dimentions: ', G.shape, '\n', 'Matrix: ', G)

    return G

unfold(X, mode=2)

mode = 0


# def tucker (rank, modes, iter):
#     X = np.arange(125).reshape(5, 5, 5)
#     print(X)
#
#
#     return G, B1, B2, B3
#
#
# tucker(rank=0, modes=0, iter=0)