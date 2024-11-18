import tensorflow as tf
import math
import itertools
import State
import PolicyState
import Definitions
import Parameters

alpha, beta, chi, delta, eta, nu  = Parameters.alpha, Parameters.beta, Parameters.chi, Parameters.delta,Parameters.eta, Parameters.nu
rho_z, sigma_z  = Parameters.rho_z, Parameters.sigma_z

# Dummy quadrature (shock =0, probability = 1)
shock_values = tf.constant([0.0])
shock_probs = tf.constant([1.0])  # Dummy probability

def total_step_random(prev_state, policy_state):
    """State dependant random shock to simulate the economy."""
    _ar = AR_step(prev_state)
    _shock = shock_step_random(prev_state)
    _policy = policy_step(prev_state, policy_state)

    _total_random = _ar + _shock + _policy
    return _total_random

def shock_step_random(prev_state):
    """Populate uncertainty for simulating the economy."""
    _random_normals = Parameters.rng.normal([prev_state.shape[0], 1])

    _shock_step = tf.zeros_like(prev_state)  # Initialization
    _shock_step = State.update(
        _shock_step, "lZt", sigma_z * _random_normals[:, 0])    
    return _shock_step


def total_step_spec_shock(prev_state, policy_state, shock_index):
    """State specific shock to evaluate the expectation operator."""
    _ar = AR_step(prev_state)
    _shock = shock_step_spec_shock(prev_state, shock_index)
    _policy = policy_step(prev_state, policy_state)

    _total_spec = _ar + _shock + _policy
    return _total_spec

def shock_step_spec_shock(prev_state, shock_index):
    """Populate uncertainty to evaluate an expectation operator."""
    _shock_step = tf.zeros_like(prev_state)  # Initialization

    # Next line doesnt do anything when shock_values = [0.0] 
    #_shock_step = State.update(
     #   _shock_step, 'lZt', sigma_z * tf.repeat(
      #      shock_values[shock_index, 0], prev_state.shape[0]))
    return _shock_step

def AR_step(prev_state):
    """AR(1) process of lZt."""
    _ar_step = tf.zeros_like(prev_state)  # Initialization
    _ar_step = State.update(_ar_step, 'lZt', rho_z * State.lZt(prev_state))
    return _ar_step


def policy_step(prev_state, policy_state):
    """Update state variables."""
    _policy_step = tf.zeros_like(prev_state)  # Initialization
    _policy_step = State.update(
        _policy_step, 'lKt', Definitions.get_lKn(prev_state, policy_state))

#     _random_uniform = Parameters.rng.uniform([prev_state.shape[0], 1])
#     _policy_step = State.update(
#         _policy_step, 'K_t',  _random_uniform[:,0] *
#                                (Parameters.k_ub - Parameters.k_lb) + Parameters.k_lb)
    return _policy_step