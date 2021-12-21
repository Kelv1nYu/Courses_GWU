#%%
import numpy as np
import matplotlib.pyplot  as plt
import scipy.linalg as la
import sympy 
from mpl_toolkits import mplot3d


#%%
# E.2 and E.3:

x = np.linspace(-15, 5, 1000)
y = 2*(x**3) + 24 * (x**2) - 54 * x

plt.title("E.2 and E.3")
plt.plot(x, y)


plt.show()

#%%
# E.4

def f1(x, y):
    return (x ** 2 + y ** 2)

x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)

X, Y = np.meshgrid(x, y)
Z = f1(X, Y)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# %%
# E.5

def f2(x, y):
    return (2*x*y + x ** 2 + y)

x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)

X, Y = np.meshgrid(x, y)
Z = f2(X, Y)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

#%%
# E.7

A = np.array([[2,0],[0,5]])
B = np.array([[5,1],[4,5]])
C = np.array([[3,5],[3,1]])

# function of getting eig_values and eig_vectors
def print_val_vec(matrix):
    eig_val, eig_vec = np.linalg.eig(matrix)
    for val in eig_val:
        print('eig_val: ', val)
    
    for vec in eig_vec:
        print('eig_vec: ', vec)


print_val_vec(A)
'''
eig_val:  2.0
eig_val:  5.0
eig_vec:  [1. 0.]
eig_vec:  [0. 1.]
'''
print_val_vec(B)
'''
eig_val:  7.0
eig_val:  3.0
eig_vec:  [ 0.4472136 -0.4472136]
eig_vec:  [0.89442719 0.89442719]
'''
print_val_vec(C)
'''
eig_val:  6.0
eig_val:  -2.0
eig_vec:  [ 0.85749293 -0.70710678]
eig_vec:  [0.51449576 0.70710678]
'''

# %%
# E.9

# function of checking linearly dependence
def check(matrix):
    # check if the rank of matrix is full rank
    # we say the square matrix is full rank if all rows and columns are linearly independent.
    if np.linalg.matrix_rank(matrix) == matrix.shape[0]:
        print('Independent')
    else:
        print('Dependent')

#%%
# E.9 - 1

V1 = np.array([[1, 1, 1],
               [2, 0, 2],
               [3, 1, 1]])

check(V1)
# rank(V1) = 3(full column and row rank), so Independent.

# Therefore, dimension is equal to the number of vectors in the basis set, 3.

#%%
# E.9 - 2

x = np.linspace(0, 2 * np.pi, 100)
y1, y2, y3 = np.sin(x), np.cos(x), np.cos(2*x)

 
plt.plot(x, y1)
plt.plot(x, y2)
plt.plot(x, y3)
 
plt.title('E.9 - 2')
plt.xlabel('x')
plt.ylabel('y')
 
plt.show()

# Three lines have intersections, so this linear system has solution - Independent.

# Therefore, dimension is equal to the number of vectors in the basis set, 3.

#%%
# E.9 - 3

t = np.linspace(0, 2 * np.pi, 100)
y1 = 1 + t
y2 = 2 - t

 
plt.plot(t, y1)
plt.plot(t, y2)
 
plt.title('E.9 - 3')
plt.xlabel('t')
plt.ylabel('y')
 
plt.show()
# Two lines have intersection, so this linear system has solution - Independent.

# Therefore, dimension is equal to the number of vectors in the basis set, 2.

#%%
# E.9 - 4

V4 = np.array([[1, 1, 3],
               [2, 0, 4],
               [2, 0, 4],
               [1, 1, 3]])

check(V4)

# rank(V4) = 2 ≠ 3(column rank), so Dependent.

# Only 2 vectors are linearly independent, therefore, dimension is equal to the number of vectors in the basis set, 2.

# %%
