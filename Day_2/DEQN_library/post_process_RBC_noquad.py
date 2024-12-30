import importlib
import pandas as pd
import tensorflow as tf
import Parameters
import matplotlib.pyplot as plt
import State
import PolicyState
import Definitions
from Graphs import run_episode
import sys
# Temporay:
import os

"""
post processing of RBC without quadrature
To Do:
1. Run a single simulation: check results
2. Plot 2D policies
3. Compute policy functions on ndgrid (in one column) & do OLS
4. Compute Euler residuals on full grid
"""


Hooks = importlib.import_module(Parameters.MODEL_NAME + ".Hooks")

import Globals
Globals.POST_PROCESSING=True

# --------------------------------------------------------------------------- #
# Simulation periods and batch size
# --------------------------------------------------------------------------- #
# Import equations
Equations = importlib.import_module(Parameters.MODEL_NAME + ".Equations")

# Number of state, policy and defined variables
N_st = len(Parameters.states)  # Number of state variables
N_ps = len(Parameters.policy_states)  # Number of policy variables


# Parameters:
alpha, beta, chi, delta, eta, nu  = Parameters.alpha, Parameters.beta, Parameters.chi, Parameters.delta,Parameters.eta, Parameters.nu
sigma_z, rho_z  = Parameters.sigma_z, Parameters.rho_z

def get_ss():
    omega = (1-beta*(1-delta))/(alpha*beta)    
    tmp = ((1-alpha)/chi*(omega-delta)**-nu)**eta
    kss = (tmp*omega**((alpha*eta+1)/(alpha-1)))**(1/(1+eta*nu))
    css = (omega-delta)*kss
    hss = omega**(1/(1-alpha))*kss
    zss = tf.constant(1, dtype=tf.float32)
    yss = zss*kss**alpha*hss**(1-alpha)
    sss = 1-css/yss
    return kss,css,hss,zss,sss

kss,css,hss,zss,sss = get_ss()
tf.print("kss (logs)",tf.math.log(kss))
tf.print("css (logs)",tf.math.log(css))
tf.print("hss (logs)",tf.math.log(hss))

##### PLOT 1D policies #######
lng = 2
bb = 5 # = nodes = batches
sim_state = tf.zeros([lng,bb,N_st]) # zeros in correct shape

# Set initial state (matrix):
lk_dev = 0.2
lk_bnds = [tf.math.log(kss)-lk_dev,tf.math.log(kss)+lk_dev]
lk_A = tf.linspace(lk_bnds[0],lk_bnds[1],bb)
lz_A = tf.fill([bb],tf.math.log(zss))
ini_A = tf.stack([lk_A,lz_A], axis=1)

sim_state_A = tf.tensor_scatter_nd_update(sim_state,[[0]], [ini_A])
sim_state_A = run_episode(sim_state_A)
#print("sim A",sim_state_A)

# Plot policy:
sim_state_A_rs = tf.reshape(sim_state_A, [lng * bb,N_st])
print("sim A reshaped",sim_state_A_rs)
ps_A_rs = Parameters.policy(sim_state_A_rs)
print("ps A reshaped",ps_A_rs)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

st_A = tf.reshape(ps_A_rs[0:bb,0],[-1])
print("lk_A",lk_A)
print("st_A",st_A)
plt.plot(lk_A,st_A)
plt.scatter(tf.math.log(kss),sss,c='red')
plt.show()

lz_fac  = 2.6 # multiple of stnd. deviation
lz_std = tf.math.sqrt(sigma_z**2 / (1-rho_z**2) )
z_nodes = 11
lz_bnds = [-lz_fac*lz_std,lz_fac*lz_std,bb]
lz_B = tf.linspace(lz_bnds[0],lz_bnds[1],bb)
lk_B = tf.fill([bb],tf.math.log(kss))
ini_B = tf.stack([lk_B,lz_B], axis=1)
sim_state_B = tf.tensor_scatter_nd_update(sim_state,[[0]], [ini_B])
sim_state_B = run_episode(sim_state_B)

# Plot policy:
sim_state_B_rs = tf.reshape(sim_state_B, [lng * bb,N_st])
print("sim B reshaped",sim_state_B_rs)
ps_B_rs = Parameters.policy(sim_state_B_rs)
print("ps B reshaped",ps_B_rs)

st_B = tf.reshape(ps_B_rs[0:bb,0],[-1])
print("lz_B",lz_B)
print("st_B",st_B)
plt.plot(lz_B,st_B)
plt.scatter(tf.math.log(zss),sss,c='red')
plt.show()

