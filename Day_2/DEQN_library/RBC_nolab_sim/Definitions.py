import tensorflow as tf
import numpy as np
import Parameters
import State
import PolicyState

alpha, beta, delta, nu  = Parameters.alpha, Parameters.beta, Parameters.delta, Parameters.nu
rho_z, sigma_z  = Parameters.rho_z, Parameters.sigma_z

def get_Kt(state,policy_state):
    """Take exponent of log(Kt)"""
    _Kt = tf.math.exp(State.lKt(state))
    return _Kt

def get_Zt(state,policy_state):
    """Take exponent of log(Zt)"""
    _Zt = tf.math.exp(State.lZt(state))
    return _Zt

# Next periods capital
def get_lKn(state, policy_state):
    """ Capital in next period, given state & policy """
    _It = get_It(state,policy_state)
    _Kt = get_Kt(state,policy_state)
    _lKn = tf.math.log(_It + (1-delta)*_Kt)
    return _lKn

# RHS of Euler in period t(!!)
def get_RHSt(state, policy_state):
    """ RHS in period t!! (given state & policy) """
    _Rt = get_Rt(state,policy_state)

    _RHSt = beta*get_marg_ut(state,policy_state)*(_Rt + 1 - delta)
    return _RHSt

def get_Ct(state,policy_state):
    """Get Ct"""
    _Yt = get_Yt(state,policy_state)
    _Ct = (1-PolicyState.st(policy_state))*_Yt
    return _Ct

def get_It(state,policy_state):
    """Get It"""
    _Yt = get_Yt(state,policy_state)
    _It = PolicyState.st(policy_state)*_Yt
    return _It

def get_marg_ut(state,policy_state):
    _Ct = get_Ct(state,policy_state)
    _dUdC = _Ct**-nu
    return _dUdC

# Model subfunctions (no state or policy)
def get_Yt(state,policy_state): 
    _Kt = get_Kt(state,policy_state)
    _Zt = get_Zt(state,policy_state)
    _Yt = _Zt*_Kt**alpha
    return _Yt#,Rt,Wt


def get_Rt(state,policy_state): 
    _Kt = get_Kt(state,policy_state)
    _Zt = get_Zt(state,policy_state)
    _Rt = alpha*_Zt*_Kt**(alpha-1)
    return _Rt
