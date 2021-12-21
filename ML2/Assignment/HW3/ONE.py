#%%
import numpy as np

#%%
# E.2

v1 = np.array([-1, 1, 0])
v2 = np.array([1, 1, -2])
v3 = np.array([1, 1, 0])

res = 0.5 * v1 - v2 + 2.5 * v3
print(res)

# res is original matrix [1. 2. 2.]


#%%
# E.5

A = np.array([[1, 1], [-1, 1]])

# find eigen values and eigen vectors of A
eig_val, eig_vec = np.linalg.eig(A)

print('eig_val: ', eig_val)
# eig_val:  [1.+1.j 1.-1.j]
print('eig_vec: ', eig_vec)
# eig_vec:  [[0.70710678+0.j         0.70710678-0.j        ]
# [0.        +0.70710678j 0.        -0.70710678j]]

# find the matrix representation for A relative to the eigenvectors as the basis vector
e=np.dot(np.dot(np.linalg.inv(eig_vec),A),eig_vec) # e = [B^-1AB]
print(e)
# [[1.+1.j 0.+0.j]
# [0.+0.j 1.-1.j]]
# %%
