# ================================================================================================================
#
# This prog. plots the max frec for varying global couplings (G)
#
# see:
# [D*2014]  Deco et al. (2014) J Neurosci.
#           http://www.jneurosci.org/content/34/23/7886.long
#
# By Gustavo Patow
#
# Optimized by Facundo Faragó
# ================================================================================================================
########### Naskar
# ============== import libraries
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from tqdm import tqdm

# ============== models
import Models.Naskar as Naskar
import WholeBrain.Models.DynamicMeanField as DMF

# ============== integrators
import WholeBrain.Integrators.EulerMaruyama as integrator
integrator.verbose = False

# ============== FIC mechanism
import WholeBrain.Utils.FIC.Balance_DecoEtAl2014 as Deco2014Mechanism
import WholeBrain.Utils.FIC.Balance_Herzog2022 as Herzog2022Mechanism
import WholeBrain.Utils.FIC.BalanceFIC as BalanceFIC
BalanceFIC.integrator = integrator

np.random.seed(42)  # Fix the seed for debug purposes...

def simulate_we(we, C, N, dt, Tmaxneuronal):
    DMF.setParms({'SC': C, 'we': we, 'J': np.ones(N)})  # Configurar parámetros
    integrator.recompileSignatures()
    v = integrator.simulate(dt, Tmaxneuronal)[:, 1, :]
    return np.max(np.mean(v, 0))

def plotMaxFrecForAllWe(C, wStart=0, wEnd=6+0.001, wStep=0.05,
                        extraTitle='', precompute=True, fileName_D=None, fileName_H=None, num_processes=1):

    ########### Naskar
    integrator.neuronalModel = Naskar
    Naskar.setParms({'SC': C})

    # Integration parms...
    dt = 0.1
    tmax = 10000.
    Tmaxneuronal = int((tmax+dt))
    # all tested global couplings (G in the paper):
    wes = np.arange(wStart + wStep, wEnd, wStep)  # warning: the range of wes depends on the conectome.
    N = C.shape[0]


    print("======================================")
    print("=    simulating Naskar               =")
    print("======================================")
    maxRateFIC = np.zeros(len(wes))
    for kk, we in enumerate(wes):  # iterate over the weight range (G in the paper, we here)
        print("\nProcessing: {}  ".format(we), end='')
        Naskar.setParms({'we': we})
        integrator.recompileSignatures()
        v = integrator.simulate(dt, Tmaxneuronal)[:,1,:]  # [1] is the output from the excitatory pool, in Hz.
        maxRateFIC[kk] = np.max(np.mean(v,0))
        print("maxRateFIC => {}".format(maxRateFIC[kk]))
    nask, = plt.plot(wes, maxRateFIC)
    nask.set_label("Naskar")


    ########### Herzog
    integrator.neuronalModel = DMF
    BalanceFIC.balancingMechanism = Herzog2022Mechanism
    DMF.setParms({'SC': C})

    print("======================================")
    print("=    simulating E-E (no FIC)         =")
    print("======================================")
    maxRateNoFIC = np.zeros(len(wes))
    DMF.setParms({'J': np.ones(N)})  # E-E = Excitatory-Excitatory, no FIC...
    for kk, we in enumerate(wes):  # iterate over the weight range (G in the paper, we here)
        print("Processing: {}".format(we), end='')
        DMF.setParms({'we': we})
        integrator.recompileSignatures()
        v = integrator.simulate(dt, Tmaxneuronal)[:, 1, :]  # [1] is the output from the excitatory pool, in Hz.
        maxRateNoFIC[kk] = np.max(np.mean(v, 0))
        print(" => {}".format(maxRateNoFIC[kk]))
    ee, = plt.plot(wes, maxRateNoFIC)
    ee.set_label("E-E")

    print("======================================")
    print("=    simulating Herzog               =")
    print("======================================")

    # Resto del código para la simulación FIC
    maxRateFIC = np.zeros(len(wes))
    if precompute:
        BalanceFIC.Balance_AllJ9(C, wes, baseName=fileName_H)
    for kk, we in enumerate(wes):
        print("\nProcessing: {}  ".format(we), end='')
        DMF.setParms({'we': we})
        balancedJ = BalanceFIC.Balance_J9(we, C, fileName_H.format(np.round(we, decimals=2)))['J'].flatten()
        integrator.neuronalModel.setParms({'J': balancedJ})
        integrator.recompileSignatures()
        v = integrator.simulate(dt, Tmaxneuronal)[:, 1, :]
        maxRateFIC[kk] = np.max(np.mean(v, 0))
        print("maxRateFIC => {}".format(maxRateFIC[kk]))
    fic_h, = plt.plot(wes, maxRateFIC)
    fic_h.set_label("Herzog")

    ########### Deco
    BalanceFIC.balancingMechanism = Deco2014Mechanism

    print("======================================")
    print("=    simulating Deco                 =")
    print("======================================")

    # Resto del código para la simulación FIC
    maxRateFIC = np.zeros(len(wes))
    if precompute:
        BalanceFIC.Balance_AllJ9(C, wes, baseName=fileName_D)
    for kk, we in enumerate(wes):
        print("\nProcessing: {}  ".format(we), end='')
        DMF.setParms({'we': we})
        balancedJ = BalanceFIC.Balance_J9(we, C, fileName_D.format(np.round(we, decimals=2)))['J'].flatten()
        integrator.neuronalModel.setParms({'J': balancedJ})
        integrator.recompileSignatures()
        v = integrator.simulate(dt, Tmaxneuronal)[:, 1, :]
        maxRateFIC[kk] = np.max(np.mean(v, 0))
        print("maxRateFIC => {}".format(maxRateFIC[kk]))
    fic_d, = plt.plot(wes, maxRateFIC)
    fic_d.set_label("Deco")

    for line, color in zip([1.47, 4.45], ['r', 'b']):
        plt.axvline(x=line, label='line at x = {}'.format(line), c=color)
    plt.title("Large-scale network (DMF) + (MDMF)" + extraTitle)
    plt.ylabel("Maximum rate (Hz)")
    plt.xlabel("Global Coupling (G = we)")
    plt.legend()
    plt.show()

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF