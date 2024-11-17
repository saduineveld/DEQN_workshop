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


Hooks = importlib.import_module(Parameters.MODEL_NAME + ".Hooks")

import Globals
Globals.POST_PROCESSING=True


# Plot policy function
alpha = Parameters.alpha
beta = Parameters.beta
Kss = (1 / (beta * alpha))**(1/(alpha - 1))
print("Kss",Kss)
Css = Kss**alpha - Kss
print("Css",Css)
lK_dev = 0.2
nodes = 15
K_bnds = tf.math.exp([tf.math.log(Kss)-lK_dev,tf.math.log(Kss)+lK_dev])
K_plot = tf.linspace(K_bnds[0],K_bnds[1], nodes)
K_plot = tf.reshape(K_plot,[nodes,1])
K_plot = tf.Variable(K_plot)
tf.print("K_plot",K_plot)
tf.print("K_plot",K_plot.shape)

# Consumption:
s_plot = Parameters.policy(K_plot)
tf.print("s_plot",s_plot)
Y_plot = K_plot ** alpha
C_plot = (1-s_plot)*Y_plot
#plt.plot(K_plot,C_plot)
#plt.show()
tf.print("C_plot",C_plot)
#os.system("pause")

# Alternative computation



tf.get_logger().setLevel('CRITICAL')

pd.set_option('display.max_columns', None)
starting_policy = Parameters.policy(Parameters.starting_state)

Equations = importlib.import_module(Parameters.MODEL_NAME + ".Equations")


# Simulate single path        
Parameters.initialize_each_episode = False        
if not Parameters.initialize_each_episode:
    ## simulate from a single starting state which is the mean across the N_sim_batch individual trajectory starting states
    simulation_starting_state = tf.math.reduce_mean(Parameters.starting_state, axis = 0, keepdims=True)
    ## simulate a long range and calculate variable bounds + means from it for plotting
    print("Running a long simulation path")
    N_simulated_episode_length = 100 #Parameters.N_episode_length or 10000
    N_simulated_batch_size = 1 # Parameters.N_batch_size or 1
        
state_episode = tf.tile(tf.expand_dims(simulation_starting_state, axis = 0), [N_simulated_episode_length, 1, 1])

print(state_episode.shape)


print("Running episode to get range of variables...")
# we are not going to re-run this graph, so let's not trace it
tf.config.experimental_run_functions_eagerly(True)

state_episode = tf.reshape(run_episode(state_episode), [N_simulated_episode_length * N_simulated_batch_size,len(Parameters.states)])

print(state_episode.shape)

mean_states = tf.math.reduce_mean(state_episode, axis = 0, keepdims=True)
min_states = tf.math.reduce_min(state_episode, axis = 0)
max_states = tf.math.reduce_max(state_episode, axis = 0)

        
print("Finished run. Calculating Euler discrepancies...")

## calculate euler deviations
policy_episode = Parameters.policy(state_episode)
euler_discrepancies = pd.DataFrame(Equations.equations(state_episode, policy_episode))
print("Euler discrepancy (absolute value) metrics")
print(euler_discrepancies.abs().describe(include='all'))

# save all relevant quantities along the trajectory 
euler_discrepancies.to_csv(Parameters.LOG_DIR + "/simulated_euler_discrepancies.csv", index=False)

state_episode_df = pd.DataFrame({s:getattr(State,s)(state_episode) for s in Parameters.states})
state_episode_df.to_csv(Parameters.LOG_DIR + "/simulated_states.csv", index=False)

policy_episode_df = pd.DataFrame({ps:getattr(PolicyState,ps)(policy_episode) for ps in Parameters.policy_states})
policy_episode_df.to_csv(Parameters.LOG_DIR + "/simulated_policies.csv", index=False)

definition_episode_df = pd.DataFrame({d:getattr(Definitions,d)(state_episode, policy_episode) for d in Parameters.definitions})
definition_episode_df.to_csv(Parameters.LOG_DIR + "/simulated_definitions.csv", index=False)

print("State metrics")
print(state_episode_df.describe(include='all'))

print("Policy metrics")
print(policy_episode_df.describe(include='all'))

print("Definition metrics")
print(definition_episode_df.describe(include='all'))

del sys.modules['Parameters']
