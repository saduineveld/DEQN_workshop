import tensorflow as tf
import Definitions as Df
import State
import PolicyState as PS
import Parameters

alpha, beta, chi, delta, eta, nu  = Parameters.alpha, Parameters.beta, Parameters.chi, Parameters.delta,Parameters.eta, Parameters.nu

def equations(state, policy_state):
    """Define the dictionary of loss functions."""

    E_t = State.E_t_gen(state, policy_state)

    Zt = Df.get_Zt(state)

    Kt = State.Kt(state)

    Ct = PS.Ct(policy_state)

    loss_dict = {}



