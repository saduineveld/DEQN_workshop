import tensorflow as tf
import Definitions
import State
import PolicyState
import Parameters

alpha, beta, chi, delta, eta, nu  = Parameters.alpha, Parameters.beta, Parameters.chi, Parameters.delta,Parameters.eta, Parameters.nu

def equations(state, policy_state):
    """Define the dictionary of loss functions."""

    # State variables (t)
    #Zt = Definitions.get_Zt(state)
    #Kt = Definitions.get_Kt(state)
    # Policy (t)
    Ct = Definitions.get_Ct(PolicyState)

    loss_dict = {}

    E_t = State.E_t_gen(state, policy_state)
    # Compute expecation of RHS of Euler equation
    RHS_Eul = E_t(lambda s, ps: Definitions.get_RHSt(s,ps))

    loss_dict['REE'] = RHS_Eul/Definitions.marg_ut(Ct) - 1

    return loss_dict

    



