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
num_st = len(Parameters.states)  # Number of state variables
num_ps = len(Parameters.policy_states)  # Number of policy variables


# Parameters:
alpha, beta, chi, delta, eta, nu  = Parameters.alpha, Parameters.beta, Parameters.chi, Parameters.delta,Parameters.eta, Parameters.nu
sigma_z, rho_z  = Parameters.sigma_z, Parameters.rho_z

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

# Plot pol. in lk and lz:
lng = 5
bb = 15 # = nodes = batches
#Create zeros [lng,bb,nn]
sim_state = tf.zeros([lng,bb,num_st])
print("Initial state",sim_state)

# Set initial state (matrix):
lk_dev = 0.2
lk_bnds = [tf.math.log(kss)-lk_dev,tf.math.log(kss)+lk_dev]
lk_A = tf.linspace(lk_bnds[0],lk_bnds[1],bb)
lz_A = tf.zeros(bb)
#lk_plot = tf.reshape(lk_plot,[bb,1])
print("lk_A",lk_A)
print("lz_A",lz_A)
lz_fac  = 2.6 # multiple of stnd. deviation
lz_std = tf.math.sqrt(sigma_z**2 / (1-rho_z**2) )
z_nodes = 11
lz_bnds = [-lz_fac*lz_std,lz_fac*lz_std,bb]
lz_B = tf.linspace(lz_bnds[0],lz_bnds[1],bb)
lk_B = tf.fill([bb],tf.math.log(kss))
#lz_plot = tf.reshape(lz_plot,[bb,1])
print("lk_B",lk_B)
print("lz_B",lz_B)

ini_A = tf.stack([lk_A,lz_A], axis=1)
print("Initial state A",ini_A)

ini_B = tf.stack([lk_B,lz_B], axis=1)


# Replace first matrix ini sim_state:
sim_state_A = tf.tensor_scatter_nd_update(sim_state,[[0]], [ini_A])
print("State A",sim_state_A)
sim_state_B = tf.tensor_scatter_nd_update(sim_state,[[0]], [ini_B])
print("State B",sim_state_B)

ini = tf.constant([Parameters.lk0,Parameters.lz0])

ini2 = tf.constant([5,3])
print("Initial state",ini2)

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
tf.print("Starting_state",simulation_starting_state_batch)

