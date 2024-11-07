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



# Next periods capital
def get_Kn(state, policy_state):
    """ Capital in next period, given state & policy """
    _Kt = State.K_t(state)
    _Zt = get_Zt(state)
    _Ct = PolicyState.C_t(policy_state)
    _Ht = get_Ht(_Zt,_Kt,_Ct)
    _Kn = prod(_Zt,_Kt,_Ht) + (1-delta)*_Kt - _Ct
    return _Kn

def get_Zt(state):
    """Take exponent of log(Zt)"""
    _Zt = tf.math.exp(lZt(state))
    return _Zt

# Model subfunctions (no state or policy)
def get_Ht(Zt,Kt,Ct):
    """Labour supply"""
    Ht = ((1-alpha/chi)*Ct**-nu*Zt*Kt**alpha)**(eta/(1+alpha*eta))
    return Ht

def marg_ut(Ct):
    dUdC = Ct**-nu
    return dUdC

def get_Ct(Zt,Kt,Ht,Kn):
    Ct = prod(Zt,Kt,Ht) + (1-delta)*Kt - Kn
    return Ct

def prod(Zt,Kt,Ht): 
    Yt = Zt*Kt**alpha*Ht**(1-alpha)
    #Rt = alpha*Yt/Kt
    #Wt = (1-alpha)*Yt/Ht
    return Yt#,Rt,Wt