import numpy as np
import scipy.linalg as sc



M = np.identity(8)

C = np.array([[2, -1, 0, 0, 0, 0, 0, 0],
             [-1, 2, -1, 0, 0, 0, 0, 0],
             [0, -1, 2, -1, 0, 0, 0, 0],
             [0, 0, -1, 2, -1, 0, 0, 0],
             [0, 0, 0, -1, 2, -1, 0, 0],
             [0, 0, 0, 0, -1, 2, -1, 0],        
             [0, 0, 0, 0, 0, -1, 2, -1],
             [0, 0, 0, 0, 0, 0, -1, 1]])        
    
K = np.array([[2000, -1000, 0, 0, 0, 0, 0, 0],
              [-1000, 2000, -1000, 0, 0, 0, 0, 0],
              [0, -1000, 2000, -1000, 0, 0, 0, 0],
              [0, 0, -1000, 2000, -1000, 0, 0, 0],
              [0, 0, 0, -1000, 2000, -1000, 0, 0],
              [0, 0, 0, 0, -1000, 2000, -1000, 0],
              [0, 0, 0, 0, 0, -1000, 2000, -1000],
              [0, 0, 0, 0, 0, 0, -1000, 1000]])



# Define the number of DOFs of the model
ndof = np.shape(M)[1]

# Define the selection matrix for the input
nf = 1
Sp = np.zeros((ndof,nf))
Sp[0][0] = 1

# Define the contrinuous-time system matrices [Ac] and [Bc]

Z = np.zeros((ndof,ndof))
I = np.identity(ndof)
Ac = np.vstack((np.hstack((Z, I)),
               (np.hstack((-np.linalg.solve(M,K), -np.linalg.solve(M,C))))))
Bc = np.vstack((Z,np.linalg.solve(I.T,M.T)))@Sp

# Define the discrete-time system matrices [A] and [B]

dt = 0.001
A = sc.expm(Ac * dt)
B = (A - np.identity(2*ndof)) @ np.linalg.inv(Ac) @ Bc


