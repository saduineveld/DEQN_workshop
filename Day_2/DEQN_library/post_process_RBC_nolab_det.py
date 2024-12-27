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

# Steady state
def get_ss():
    omega = (1-beta*(1-delta))/(alpha*beta)    
    kss = omega**(1/(alpha-1))
    css = kss**alpha-delta*kss
    sss = 1 - css/(kss**alpha)
    return kss,css,sss

kss,css,sss = get_ss()

# Plot policy function
print("kss",kss)
print("css",css)
print("sss",sss)

lK_dev = 0.2
nodes = 15
K_bnds = tf.math.exp([tf.math.log(kss)-lK_dev,tf.math.log(kss)+lK_dev])
K_plot = tf.linspace(K_bnds[0],K_bnds[1], nodes)
K_plot = tf.reshape(K_plot,[nodes,1])
print("K_plot",K_plot)
tf.print("K_plot",K_plot.shape)

# Consumption:
s_plot = Parameters.policy(K_plot)
print("s_plot",s_plot)

plt.plot(K_plot,s_plot)
plt.scatter(kss,sss,c='red')
plt.show()


Y_plot = K_plot ** alpha
C_plot = (1-s_plot)*Y_plot
#plt.plot(K_plot,C_plot)
#plt.show()
tf.print("C_plot",C_plot)
#os.system("pause")
plt.plot(K_plot,C_plot)
plt.scatter(kss,css,c='red')
plt.show()

### Sun simulation from different starting states ###
lng = 10#lenght of simulation
batches = 5#number of paths

# Set state-array (pre-allocate lng x batches x state variables) + set initial values of first column vector (note row indices are second dimension, hence inidices [[0]])
sim_state = tf.zeros([lng,batches,1])
ini_state = tf.linspace(K_bnds[0],K_bnds[1],batches)#1 dim vector
ini_state = tf.cast(ini_state,tf.float32)#Change type from float64 to float32
ini_state = tf.reshape(ini_state,[1,batches,1])
indices_split = tf.constant([[0]])
sim_state = tf.tensor_scatter_nd_update(sim_state,indices_split,ini_state)
#print("Input of simulation:",sim_state)

# Get simulation of state variables
sim_state = run_episode(sim_state)
print("Output of simulation:",sim_state)

# Reshape in 2D matrix (for each state variable, but there's only one here):
sim_k = tf.reshape(sim_state,[-1,batches])
print("State in 2d matrix",sim_k)
sim_k_df = pd.DataFrame(sim_k)
sim_k_df.to_csv(Parameters.LOG_DIR + "/sim_k.csv", index=False)

