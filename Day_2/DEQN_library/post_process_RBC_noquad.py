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

# Parameters:
alpha, beta, chi, delta, eta, nu  = Parameters.alpha, Parameters.beta, Parameters.chi, Parameters.delta,Parameters.eta, Parameters.nu

def get_ss():
    omega = (1-beta*(1-delta))/(alpha*beta)    
    tmp = ((1-alpha)/chi*(omega-delta)**-nu)**eta
    kss = (tmp*omega**((alpha*eta+1)/(alpha-1)))**(1/(1+eta*nu))
    css = (omega-delta)*kss
    hss = omega**(1/(1-alpha))*kss
    return kss,css,hss

kss,css,hss = get_ss()

print("kss",kss)
print("css",css)
print("hss",hss)

# Note states are in logs {lK,lZ}
lK_dev = 0.2
nodes = 15
lK_bnds = [tf.math.log(kss)-lK_dev,tf.math.log(kss)+lK_dev]
lK_plot = tf.linspace(lK_bnds[0],lK_bnds[1], nodes)
lK_plot = tf.reshape(lK_plot,[nodes,1])
print("lK_plot",lK_plot)
tf.print("lK_plot",lK_plot.shape)