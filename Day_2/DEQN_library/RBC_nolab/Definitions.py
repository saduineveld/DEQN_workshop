import tensorflow as tf
import numpy as np
import Parameters
import State
import PolicyState

alpha, beta, delta, nu  = Parameters.alpha, Parameters.beta, Parameters.delta, Parameters.nu
rho_z, sigma_z  = Parameters.rho_z, Parameters.sigma_z

# Next periods capital
def get_Kn(state, policy_state):
    """ Capital in next period, given state & policy """
    _It = get_It(state,policy_state)
    _Kn = _It + (1-delta)*State.Kt(state)
    return _Kn

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
    _Yt = State.Zt(state)*State.Kt(state)**alpha
    return _Yt#,Rt,Wt


def get_Rt(state,policy_state): 
    _Rt = alpha*State.Zt(state)*State.Kt(state)**(alpha-1)
    return _Rt
