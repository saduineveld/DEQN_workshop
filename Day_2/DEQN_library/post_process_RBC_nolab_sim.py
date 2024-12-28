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
Example file for postprocessing 
"""
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

Hooks = importlib.import_module(Parameters.MODEL_NAME + ".Hooks")

import Globals
Globals.POST_PROCESSING=True


alpha, beta, delta, nu  = Parameters.alpha, Parameters.beta, Parameters.delta, Parameters.nu
sigma_z, rho_z  = Parameters.sigma_z, Parameters.rho_z

# Number of state, policy and defined variables
N_st = len(Parameters.states)  # Number of state variables
N_ps = len(Parameters.policy_states)  # Number of policy variables

# Steady state
def get_ss():
    omega = (1-beta*(1-delta))/(alpha*beta)    
    kss = omega**(1/(alpha-1))
    css = kss**alpha-delta*kss
    sss = 1 - css/(kss**alpha)
    zss = tf.constant(1, dtype=tf.float32)
    return kss,css,sss,zss

kss,css,sss,zss = get_ss()


# Plot policy function
print("kss",kss)
print("css",css)
print("sss",sss)

##### PLOT 1D policies #######
lng = 2
bb = 5 # = nodes = batches
sim_state = tf.zeros([lng,bb,N_st]) # zeros in correct shape

# Set initial state (matrix):
lk_dev = 0.2
lk_bnds = [-lk_dev + tf.math.log(kss),lk_dev+tf.math.log(kss)]
lk_A = tf.linspace(lk_bnds[0],lk_bnds[1],bb)
lz_A = tf.fill([bb],tf.math.log(zss))
ini_A = tf.stack([lk_A,lz_A], axis=1)
sim_state_A = tf.tensor_scatter_nd_update(sim_state,[[0]], [ini_A])
sim_state_A = run_episode(sim_state_A)

# Plot policy:
sim_state_A_rs = tf.reshape(sim_state_A, [lng * bb,N_st])
print("sim A reshaped",sim_state_A_rs)
ps_A_rs = Parameters.policy(sim_state_A_rs)
print("ps A reshaped",ps_A_rs)

st_A = tf.reshape(ps_A_rs[0:bb,0],[-1])
print("lk_A",lk_A)
print("st_A",st_A)
plt.plot(lk_A,st_A)
plt.scatter(tf.math.log(kss),sss,c='red')
plt.show()

lz_fac  = 2.6 # multiple of stnd. deviation
lz_std = tf.math.sqrt(sigma_z**2 / (1-rho_z**2) )
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
