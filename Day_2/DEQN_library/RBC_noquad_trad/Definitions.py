import tensorflow as tf
import numpy as np
import Parameters
import State
import PolicyState

alpha, beta, chi, delta, eta, nu  = Parameters.alpha, Parameters.beta, Parameters.chi, Parameters.delta,Parameters.eta, Parameters.nu

# Steady state
#def get_ss():
#    omega = (1-beta*(1-delta))/(alpha*beta)    
#    tmp = ((1-alpha)/chi*(omega-delta)**-nu)**eta
#    kss = (tmp*omega**((alpha*eta+1)/(alpha-1)))**(1/(1+eta*nu))
#    css = (omega-delta)*kss
#    hss = omega**(1/(1-alpha))*kss
#    return kss,css,hss

# Next periods capital
def get_lKn(state, policy_state):
    """ Capital in next period, given state & policy """
    _Kt = get_Kt(state,policy_state)
    _Ct = get_Ct(state,policy_state)
    _Yt = get_Yt(state,policy_state)

    _lKn = tf.math.log(_Yt + (1-delta)*_Kt - _Ct)
    return _lKn

# RHS of Euler in period t(!!)
def get_RHSt(state, policy_state):
    """ RHS in period t!! (given state & policy) """
    _Rt = get_Rt(state,policy_state)

    _RHSt = beta*get_marg_ut(state,policy_state)*(_Rt + 1 - delta)
    return _RHSt


def get_Kt(state,policy_state):
    """Take exponent of log(Kt)"""
    _Kt = tf.math.exp(State.lKt(state))
    return _Kt

def get_Zt(state,policy_state):
    """Take exponent of log(Zt)"""
    _Zt = tf.math.exp(State.lZt(state))
    return _Zt

def get_Ct(state,policy_state):
    """Take exponent of log(Ct)"""
    _Ct = tf.math.exp(PolicyState.lCt(policy_state))
    return _Ct

def get_marg_ut(state,policy_state):
    _Ct = get_Ct(state,policy_state)
    _dUdC = _Ct**-nu
    return _dUdC


# Model subfunctions (no state or policy)
def get_Ht(state,policy_state):
    """Labour supply"""
    _Kt = get_Kt(state,policy_state)
    _Zt = get_Zt(state,policy_state)
    _Ct = get_Ct(state,policy_state)

    _Ht = (((1-alpha)/chi)*_Ct**-nu*_Zt*_Kt**alpha)**(eta/(1+alpha*eta))
    return _Ht

def get_Yt(state,policy_state): 
    _Kt = get_Kt(state,policy_state)
    _Zt = get_Zt(state,policy_state)

    _Ht = get_Ht(state,policy_state)

    _Yt = _Zt*_Kt**alpha*_Ht**(1-alpha)
    return _Yt#,Rt,Wt

def get_Rt(state,policy_state): 
    _Kt = get_Kt(state,policy_state)
    _Yt = get_Yt(state,policy_state)

    _Rt = alpha*_Yt/_Kt

    return _Rt

#def get_Ct(Zt,Kt,Ht,Kn):
#    Ct = prod(Zt,Kt,Ht) + (1-delta)*Kt - Kn
#    return Ct