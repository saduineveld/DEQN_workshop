import tensorflow as tf
import math
import itertools
import State
import PolicyState
import Definitions
import Parameters

alpha, beta, chi, delta, eta, nu  = Parameters.alpha, Parameters.beta, Parameters.chi, Parameters.delta,Parameters.eta, Parameters.nu

# Probability of a dummy shock
shock_probs = tf.constant([1.0])  # Dummy probability