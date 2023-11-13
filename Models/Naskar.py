# ==========================================================================
# ==========================================================================
# ==========================================================================
# Dynamic Mean Field (DMF) model (a.k.a., Reduced Wong-Wang), from
#
# [Deco_2014] G. Deco, A. Ponce-Alvarez, P. Hagmann, G.L. Romani, D. Mantini, M. Corbetta
#             How local excitation-inhibition ratio impacts the whole brain dynamics
#             J. Neurosci., 34 (2014), pp. 7886-7898
# ==========================================================================
import numpy as np
from numba import jit
import WholeBrain.Utils.FIC.Balance_Herzog2022 as Herzog

print("Going to use the Dynamic Mean Field (DMF) neuronal model...")


def recompileSignatures():
    # Recompile all existing signatures. Since compiling isn’t cheap, handle with care...
    # However, this is "infinitely" cheaper than all the other computations we make around here ;-)
    phie.recompile()
    phii.recompile()
    dfun.recompile()

# ==========================================================================
# ==========================================================================
# ==========================================================================
# Dynamic Mean Field (DMF) Model Constants
# --------------------------------------------------------------------------
B_e = 0.0066 #ms^-1
B_i = 0.18 #ms^-1
t_glu = 100
t_gaba = 10
alfa_e = 0.072
alfa_i = 0.53
J_NMDA = 0.15       # [nA] NMDA current
I0 = 0.382  #.397  # [nA] overall effective external input
Jexte = 1.
Jexti = 0.7
w = 1.4
we = 2.1        # Global coupling scaling (G in the paper)
SC = None       # Structural connectivity (should be provided externally)
p = 3

# transfer WholeBrain:
# --------------------------------------------------------------------------
# transfer function: excitatory
ae = 310.  # [nC^{-1}], g_E in the paper
be = 125.  # = g_E * I^{(E)_{thr}} in the paper = 310 * .403 [nA] = 124.93
de = 0.16
@jit(nopython=True)
def phie(x):
    # in the paper this was g_E * (I^{(E)_n} - I^{(E)_{thr}})
    # Here, we distribute as g_E * I^{(E)_n} - g_E * I^{(E)_{thr}}, thus...
    y = (ae*x-be)
    # if (y != 0):
    return y/(1.-np.exp(-de*y))
    # else:
    #     return 0

# transfer function: inhibitory
ai = 615.  # [nC^{-1}], g_I in the paper
bi = 177.  # = g_I * I^{(I)_{thr}} in the paper = 615 * .288 [nA] = 177.12
di = 0.087
@jit(nopython=True)
def phii(x):
    # in the paper this was g_I * (I^{(I)_n} - I^{(I)_{thr}}).
    # Apply same distributing as above...
    y = (ai*x-bi)
    # if (y != 0):
    return y/(1.-np.exp(-di*y))
    # else:
    #     return 0

# transfer WholeBrain used by the simulation...
He = phie
Hi = phii

# --------------------------------------------------------------------------
# Simulation variables
# @jit(nopython=True)
def initSim(N):
    sn = 0.001 * np.ones(N)  # Initialize sn (S^E in the paper)
    sg = 0.001 * np.ones(N)  # Initialize sg (S^I in the paper)
    J = 1 * np.ones(N)
    return np.stack((sn,sg,J))


# J = None    # WARNING: In general, J must be initialized outside!
# def initJ(N):  # A bit silly, I know...
#     global J
#     J = np.ones(N)


# --------------------------------------------------------------------------
# Variables of interest, needed for bookkeeping tasks...
# xn = None  # xn, excitatory synaptic activity
# rn = None  # rn, excitatory firing rate
# @jit(nopython=True)
def numObsVars():  # Returns the number of observation vars used, here xn and rn
    return 3


# --------------------------------------------------------------------------
# Set the parameters for this model
def setParms(modelParms):
    global we, SC
    if 'we' in modelParms or 'G' in modelParms:  # I've made this mistake too many times...
        we = modelParms['we']
    if 'SC' in modelParms:
        SC = modelParms['SC']


def getParm(parmList):
    if 'we' in parmList or 'G' in parmList:  # I've made this mistake too many times...
        return we
    if 'be' in parmList:
        return be
    if 'ae' in parmList:
        return ae
    if 'SC' in parmList:
        return SC
    return None


# ----------------- Dynamic Mean Field (a.k.a., reducedWongWang) ----------------------
@jit(nopython=True)
def dfun(simVars, I_external):
    # global xn, rn
    sn = simVars[0]; sg = simVars[1]; J = simVars[2]
    xn = I0 * Jexte + w * J_NMDA * sn + we * J_NMDA * (SC @ sn) - J * sg + I_external  # Eq for I^E (5). I_external = 0 => resting state condition.
    xg = 0.7*I0 * Jexti + J_NMDA * sn - sg  # Eq for I^I (6). \lambda = 0 => no long-range feedforward inhibition (FFI)
    rn = He(xn)  # Calls He(xn). r^E = H^E(I^E) in the paper (7)
    rg = Hi(xg)  # Calls Hi(xg). r^I = H^I(I^I) in the paper (8)
    dsn = -sn*B_e + (1. - sn) * t_glu*alfa_e * rn
    dsg = -sg*B_i + rg * t_gaba*alfa_i
    dJ = rg*(rn-p)
    return np.stack((dsn, dsg, dJ)), np.stack((xn, rn, J))


# ==========================================================================
# ==========================================================================
# ==========================================================================
