# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 20:41:36 2021

@author: Victor Flores Terrazas at 
        The Hong Kong University of Science and Technology

        Python implementation of original Matlab code from:
        * Sedehi, Omid, et al. "Sequential Bayesian estimation of 
          state and input in dynamical systems using output-only measurements." 
          Mechanical Systems and Signal Processing 131 (2019): 659-688.

        Implementation of Augmented Kalman Filter originally from Matlab code
        created by O. Sedehi, based on:
        * Lourens, E., et al. "An augmented Kalman filter for force 
          identification in structural dynamics." 
          Mechanical Systems and Signal Processing 27 (2012): 446-460.
"""

import numpy as np
import scipy.linalg as sc
import matplotlib.pyplot as plt

# ---------- Structural Model ---------------------------

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

# Define the continuous-time system matrices [Ac] and [Bc]

Z = np.zeros((ndof,ndof))
I = np.identity(ndof)
Ac = np.vstack((np.hstack((Z, I)),
               (np.hstack((-np.linalg.solve(M,K), -np.linalg.solve(M,C))))))
Bc = np.vstack((Z,np.linalg.solve(I.T,M.T))) @ Sp

# Define the discrete-time system matrices [A] and [B]

dt = 0.001
A = sc.expm(Ac * dt)
B = (A - np.identity(2*ndof)) @ np.linalg.inv(Ac) @ Bc

# ---------- Input ---------------------------

np.random.seed(1)

T = 50
Ndata = T/dt + 1
t = np.linspace(0, T, num=int(Ndata))
input_mu , input_std = 0 , 5
p_real = np.random.normal(input_mu, input_std, size=(int(Ndata),int(nf)))

# ---------- Structural Responses ---------------------------

Ge = np.vstack((np.hstack((I,Z)),
               (np.hstack((Z, I))),
               (np.hstack((-np.linalg.solve(M,K), -np.linalg.solve(M,C))))))
Je = np.vstack((Z,Z,np.linalg.solve(I.T,M.T))) @ Sp

# Pre-allocate matrices
x_real = np.zeros((2*ndof,int(Ndata)))
y_real = np.zeros((3*ndof,int(Ndata)))

# Initialize
x_real[:,0] = np.zeros((1,2*ndof))
y_real[:,0] = Ge @ x_real[:,0] + Je @ p_real[0,:]

for i in range(int(Ndata)-1):
    x_real[:,i+1] = A @ x_real[:,i] + B @ p_real[i,:]
    y_real[:,i+1] = Ge @ x_real[:,i] + Je @ p_real[i,:]
    

# ---------- Measurements ---------------------------
# np.random.seed(12)

na = 2
nv = 0
nd = 0
Sa = np.zeros((na,ndof))
Sa[0][0] = 1
Sa[1][3] = 1
G = Sa @ np.hstack([-np.linalg.solve(M,K), -np.linalg.solve(M,C)])
J = Sa @ np.linalg.solve(I.T,M.T) @ Sp

ry = np.var(y_real[19,:], axis = 0) * 0.01**2

nm = na + nv + nd

R = np.identity(nm) * ry

y_meas = np.zeros((int(nm),int(Ndata)))
meas_noise = np.random.normal(0,np.sqrt(ry),(int(nm),int(Ndata)))

for i in range(int(Ndata)-1):
    y_meas[:,i] = G @ x_real[:,i] + J @ p_real[i,:] + meas_noise[:,i]
    
    
# --------- Bayesian JISE -----------------

nu = 0
kappa = 1

Qx_bay = np.identity(int(2*ndof))*10**-3

x_bay = np.zeros((int(2*ndof),int(Ndata)))
p_bay = np.zeros((int(nf),int(Ndata)))
Px_bay = np.zeros((int(2*ndof),int(2*ndof),int(Ndata)))
Pp_bay = np.zeros((int(nf),int(nf),int(Ndata)))
Pxp_bay = np.zeros((int(2*ndof),int(nf),int(Ndata)))

Px_bay[:,:,0] = np.identity(Px_bay.shape[0])*Qx_bay
Pp_bay[:,:,0] = np.identity(Pp_bay.shape[0])*10**-3

mp = np.zeros((int(nf),int(Ndata)))
mpp = np.zeros((int(nf),int(Ndata)))
mx = np.zeros((int(2*ndof),int(Ndata)))
m = np.zeros((int(nm),int(Ndata)))

for i in range(int(Ndata)-1):
    # Time update for input and state
    Pp_bay[:,:,i+1] = Pp_bay[:,:,i]
    x_bay[:,i+1] = A @ x_bay[:,i]
    Px_bay[:,:,i+1] = ((A @ Px_bay[:,:,i] @ A.T) + 
                       (B @ Pp_bay[:,:,i] @ B.T) +
                       (A @ Pxp_bay[:,:,i] @ B.T) + 
                       (B @ Pxp_bay[:,:,i].T @ A.T) +
                       Qx_bay)
    
    # Kalman Gain matirx for input estimation
    Kp = Pp_bay[:,:,i+1] @ J.T @ (np.linalg.inv(J @ Pp_bay[:,:,i+1] @ J.T + R))
    
    # Measurement update for input
    p_bay[:,i+1] = Kp @ (y_meas[:,i+1] - G @ x_bay[:,i+1])
    Pp_bay[:,:,i+1] = (Pp_bay[:,:,i+1] + 
                       Kp @ G @ Px_bay[:,:,i+1] @ G.T @ Kp.T -
                       Kp @ J @ Pp_bay[:,:,i+1])
    
    # Updating the state based on the new input
    x_bay[:,i+1] = A @ x_bay[:,i] + B @ p_bay[:,i]
    Px_bay[:,:,i+1] = ((A @ Px_bay[:,:,i] @ A.T) + 
                       (B @ Pp_bay[:,:,i] @ B.T) +
                       (A @ Pxp_bay[:,:,i] @ B.T) + 
                       (B @ Pxp_bay[:,:,i].T @ A.T) +
                       Qx_bay)
    
    # Kalman Gain for state estimation
    Kx = Px_bay[:,:,i+1] @ G.T @ np.linalg.inv(G @ Px_bay[:,:,i+1] @ G.T + R)
    
    # Measurement update for state
    x_bay[:,i+1] = (x_bay[:,i+1] + 
                    Kx @ (y_meas[:,i+1] -
                          G @ x_bay[:,i+1] -
                          J @ p_bay[:,i+1]))
    Px_bay[:,:,i+1] = (Px_bay[:,:,i+1] + 
                       Kx @ J @ Pp_bay[:,:,i+1] @ J.T @ Kx.T -
                       Kx @ G @ Px_bay[:,:,i+1])
    
    # Cross-covariance matrix
    Pxp_bay[:,:,i+1] = -Kx @ J @ Pp_bay[:,:,i+1]
    
    # Noise estimation (Bayesian)
    nu = nu + 1
    kappa = kappa + 1
    mx[:,i+1] = x_bay[:,i+1] - x_bay[:,i]
    Qx_bay = ((Qx_bay * (nu + nf)) + (mx[:,i+1] @ mx[:,i+1].T)) / (nu + nf + 1)
    m[:,i+1] = (y_meas[:,i+1] - 
                G @ x_bay[:,i+1] - 
                J @ p_bay[:,i+1])
    R = ((R*(nu + nf)) + (m[:,i+1] @ m[:,i+1].T)) / (nu + nf + 1)


# Reconstruct the signal

y_bay = np.zeros((int(3*ndof),int(Ndata)))

for i in range(int(Ndata)-1):
    y_bay[:,i] = Ge @ x_bay[:,i] + Je @ p_bay[:,i]



# ---------- AKF ---------------------------

A_akf = np.vstack((np.hstack((A,B)),
                    (np.hstack((np.zeros((nf,2*ndof)),np.identity(nf))))))
G_akf = np.hstack((G,J))

Ge_akf = np.hstack((Ge,Je))

Ac = np.vstack((np.hstack((Ac,Bc)),
                    (np.hstack((np.zeros((nf,2*ndof)),np.identity(nf))))))

Qp = np.identity(nf)*10**4
Qx_akf = np.zeros((2*ndof,2*ndof))
Qx_akf[int(ndof):,int(ndof):] = np.identity(ndof)*10**-8
Z_akf = np.zeros((2*ndof,nf))
Q_akf = np.vstack((np.hstack((Qx_akf,Z_akf)),
                   np.hstack((Z_akf.T,Qp))))

# Pre-allocate AKF state matrix
x_akf = np.zeros((int(2*ndof+nf),int(Ndata)))

# Pre-allocate Covariance 
Px_akf = np.zeros((int(2*ndof+nf),int(2*ndof+nf),int(Ndata)))

# Initialize 
Px_akf[:,:,0] = Q_akf

# AKF algorithm
for i in range(int(Ndata)-1):
    # Time update for state
    x_akf[:,i+1] = A_akf @ x_akf[:,i]
    Px_akf[:,:,i+1] = A_akf @ Px_akf[:,:,i] @ A_akf.T + Q_akf
    # Kalman Gain for state estimation
    KG_akf = Px_akf[:,:,i+1] @ G_akf.T @ np.linalg.inv(G_akf @ Px_akf[:,:,i+1] @ G_akf.T + R)
    # Measurement update for state
    x_akf[:,i+1] = x_akf[:,i+1] + (KG_akf @ (y_meas[:,i+1] - G_akf @ x_akf[:,i+1]))
    Px_akf[:,:,i+1] = Px_akf[:,:,i+1] - (KG_akf @ G_akf @ Px_akf[:,:,i+1])

# Reconstruct the signal

# Define the variable y_akf
y_akf = np.zeros((int(3*ndof),int(Ndata)))

for i in range(int(Ndata)-1):
    y_akf[:,i] = Ge_akf @ x_akf[:,i]

p_akf = x_akf[int(2*ndof):,:]


#-------- Plots ------------------------------------

plt.close(fig='all')

plot_dof = int(21) # Define the DOF to be plotted below. DOFs [17:24] are acclerations

plt.figure(0)
plt.plot(t*dt,y_real[plot_dof,:])
plt.plot(t*dt,y_akf[plot_dof,:])
plt.plot(t*dt,y_bay[plot_dof,:])
plt.xlabel('time (s)')
plt.ylabel('acceleration (m/s^2)')
plt.legend(['real','akf','bayesian'])


plt.figure(1)
plt.plot(t*dt,p_real[:,0])
plt.plot(t*dt,p_akf[0,:])
plt.plot(t*dt,p_bay[0,:])
plt.xlabel('time (s)')
plt.ylabel('displacement (m)')
plt.legend(['real','akf','bayesian'])





    
