import tensorflow as tf
import Definitions
import State
import PolicyState
import Parameters

alpha, beta, delta, nu  = Parameters.alpha, Parameters.beta, Parameters.delta, Parameters.nu
rho_z, sigma_z  = Parameters.rho_z, Parameters.sigma_z

def equations(state, policy_state):
    """Define the dictionary of loss functions."""

    E_t = State.E_t_gen(state, policy_state)

    # State variables (t)
    #Zt = Definitions.get_Zt(state)
    #Kt = Definitions.get_Kt(state)
    # Policy (t)
    #Ct = Definitions.get_Ct(state,policy_state)

    loss_dict = {}

    
    # Compute expecation of RHS of Euler equation
    RHS_Eul = E_t(lambda s, ps: Definitions.get_RHSt(s,ps) )


    #loss_dict['REE'] = RHS_Eul/Definitions.marg_ut(Ct) - 1
    loss_dict['REE'] = RHS_Eul/Definitions.get_marg_ut(state,policy_state) - 1

    return loss_dict