import tensorflow as tf
import numpy as np
import Parameters
import State
import PolicyState

alpha, beta, chi, delta, eta, nu  = Parameters.alpha, Parameters.beta, Parameters.chi, Parameters.delta,Parameters.eta, Parameters.nu

# Steady state
def get_ss():
    omega = (1-beta*(1-delta))/(alpha*beta)    
    tmp = ((1-alpha)/chi*(omega-delta)**-nu)**eta
    kss = (tmp*omega**((alpha*eta+1)/(alpha-1)))**(1/(1+eta*nu))
    css = (omega-delta)*kss
    hss = omega**(1/(1-alpha))*kss
    return kss,css,hss

# Next periods capital
def get_Kn(state, policy_state):
    """ Capital in next period, given state & policy """
    _Kt = get_Kt(state)
    _Zt = get_Zt(state)
    _Ct = get_Ct(policy_state)

    _Ht = get_Ht(_Zt,_Kt,_Ct)
    _Yt = get_Yt(_Zt,_Kt,_Ht)

    _Kn = _Yt + (1-delta)*_Kt - _Ct
    return _Kn

# RHS of Euler in period t(!!)
def get_RHSt(state, policy_state):
    """ RHS in period t!! (given state & policy) """
    _Kt = get_Kt(state)
    _Zt = get_Zt(state)
    _Ct = get_Ct(policy_state)

    _Ht = get_Ht(_Zt,_Kt,_Ct)
    _Yt = get_Yt(_Zt,_Kt,_Ht)

    _Rt = get_Rt(_Yt,_Kt)
    _RHSt = beta*marg_ut(_Ct)*(_Rt + 1 - delta)
    return _RHSt


def get_Kt(state):
    """Take exponent of log(Kt)"""
    _Kt = tf.math.exp(State.lKt(state))
    return _Kt

def get_Zt(state):
    """Take exponent of log(Zt)"""
    _Zt = tf.math.exp(State.lZt(state))
    return _Zt

def get_Ct(policy_state):
    """Take exponent of log(Ct)"""
    _Ct = tf.math.exp(PolicyState.lCt(policy_state))
    return _Ct




# Model subfunctions (no state or policy)
def get_Ht(Zt,Kt,Ct):
    """Labour supply"""
    Ht = ((1-alpha/chi)*Ct**-nu*Zt*Kt**alpha)**(eta/(1+alpha*eta))
    return Ht

def marg_ut(Ct):
    dUdC = Ct**-nu
    return dUdC

def get_Yt(Zt,Kt,Ht): 
    Yt = Zt*Kt**alpha*Ht**(1-alpha)
    #
    #Wt = (1-alpha)*Yt/Ht
    return Yt#,Rt,Wt

def get_Rt(Yt,Kt):
    _Rt = alpha*Yt/Kt
    return _Rt

#def get_Ct(Zt,Kt,Ht,Kn):
#    Ct = prod(Zt,Kt,Ht) + (1-delta)*Kt - Kn
#    return Ct