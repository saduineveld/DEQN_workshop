import Parameters
import PolicyState
import State

alpha, beta, chi, delta, eta, nu  = Parameters.alpha, Parameters.beta, Parameters.chi, Parameters.delta,Parameters.eta, Parameters.nu

# Steady state
def get_ss():
    omega = (1-beta*(1-delta))/(alpha*beta)    
    tmp = ((1-alpha)/chi*(omega-delta)**-nu)**eta
    kss = (tmp*omega**((alpha*eta+1)/(alpha-1)))**(1/(1+eta*nu))
    css = (omega-delta)*kss
    hss = omega**(1/(1-alpha))*kss
    return kss,css,hss

def H_t(state, policy_state):
    """compute output today"""
    _K_t = State.K_t(state)
    _C_t = PolicyState.C_t(policy_state)
    _Y_t = _K_t ** alpha
    return _Y_t


# Model subfunctions
def marg_ut(cc):
    dudc = cc**-nu
    return dudc

def labour(zt,kt,ct):
    ht = ((1-alpha/chi)*ct**-nu*zt*kt**alpha)**(eta/(1+alpha*eta))
    return ht

def knext(zt,kt,ht,ct):
    kn = prod(alpha,zt,kt,ht)[0] + (1-delta)*kt - ct
    return kn

def cons(azt,kt,ht,kn):
    ct = prod(alpha,zt,kt,ht) + (1-delta)*kt - kn
    return ct

def prod(zt,kt,ht): 
    yt = zt*kt**alpha*ht**(1-alpha)
    rt = alpha*yt/kt
    wt = (1-alpha)*yt/ht
    return yt,rt,wt