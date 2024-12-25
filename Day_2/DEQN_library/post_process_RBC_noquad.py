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
N_episode_length = 500

# Number of simulation batch, it should be arbitrary big enough
N_sim_batch = 10  # For testing (for distributions: use 10000)

# Import equations
Equations = importlib.import_module(Parameters.MODEL_NAME + ".Equations")

# Number of state, policy and defined variables
N_state = len(Parameters.states)  # Number of state variables
N_policy_state = len(Parameters.policy_states)  # Number of policy variables


# Parameters:
alpha, beta, chi, delta, eta, nu  = Parameters.alpha, Parameters.beta, Parameters.chi, Parameters.delta,Parameters.eta, Parameters.nu
sigma_z, rho_z  = Parameters.sigma_z, Parameters.rho_z

# Starting state (one value for each state variable only)
starting_state = tf.reshape(tf.constant([
    Parameters.lk0,Parameters.lz0]), shape=(1, N_state))

# Simulate the economy for N_episode_length time periods
simulation_starting_state = tf.tile(tf.expand_dims(
    starting_state, axis=0), [N_episode_length, 1, 1])
tf.print("Shape of sim._starting_state",simulation_starting_state.shape)

simulation_starting_state_batch = tf.tile(tf.expand_dims(
    starting_state, axis=0), [N_episode_length, N_sim_batch, 1])
tf.print("Shape of sim._starting_state_batch",simulation_starting_state_batch.shape)


def get_ss():
    omega = (1-beta*(1-delta))/(alpha*beta)    
    tmp = ((1-alpha)/chi*(omega-delta)**-nu)**eta
    kss = (tmp*omega**((alpha*eta+1)/(alpha-1)))**(1/(1+eta*nu))
    css = (omega-delta)*kss
    hss = omega**(1/(1-alpha))*kss
    zss = 1
    return kss,css,hss,zss

kss,css,hss,zss = get_ss()

print("kss",kss)
print("css",css)
print("hss",hss)

# Note states are in logs {lK,lZ}
# Construct grid vectors (keep lz=0 for now)
k_nodes = 15
lk_dev = 0.2
lk_bnds = [tf.math.log(kss)-lk_dev,tf.math.log(kss)+lk_dev]
#lk_plot = tf.linspace(lk_bnds[0],lk_bnds[1],k_nodes)
#lk_plot = tf.reshape(lk_plot,[k_nodes,1])
 
lz_fac  = 2.6 # multiple of stnd. deviation
lz_std = tf.math.sqrt(sigma_z**2 / (1-rho_z**2) )
z_nodes = 11
lz_bnds = [-lz_fac*lz_std,lz_fac*lz_std,z_nodes]