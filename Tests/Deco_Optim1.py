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
#
# ============== import libraries
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import multiprocessing
from tqdm import tqdm

# ============== chose a model
import WholeBrain.Models.DynamicMeanField as DMF

# ============== chose and setup an integrator
import WholeBrain.Integrators.EulerMaruyama as integrator
integrator.neuronalModel = DMF
integrator.verbose = False

# ============== chose a FIC mechanism
import WholeBrain.Utils.FIC.BalanceFIC as BalanceFIC
BalanceFIC.integrator = integrator
# import WholeBrain.Utils.FIC.Balance_DecoEtAl2014 as Deco2014Mechanism
# BalanceFIC.balancingMechanism = Deco2014Mechanism  # default behaviour for this project


np.random.seed(42)  # Fix the seed for debug purposes...


def simulate_we(we, C, N, dt, Tmaxneuronal):
    DMF.setParms({'SC': C, 'we': we, 'J': np.ones(N)})  # Configurar parámetros
    integrator.recompileSignatures()
    v = integrator.simulate(dt, Tmaxneuronal)[:, 1, :]
    return np.max(np.mean(v, 0))

def plotMaxFrecForAllWe(C, wStart=0, wEnd=6 + 0.001, wStep=0.05,
                        extraTitle='', precompute=True, fileName=None, num_processes=1):
    # Integration parms...
    dt = 0.1
    tmax = 10000.
    Tmaxneuronal = int((tmax + dt))
    # all tested global couplings (G in the paper):
    wes = np.arange(wStart + wStep, wEnd, wStep)  # warning: the range of wes depends on the connectome.
    N = C.shape[0]

    DMF.setParms({'SC': C})

    print("======================================")
    print("=    simulating E-E (no FIC)         =")
    print("======================================")
    maxRateNoFIC = []  # Cambia a una lista vacía

    with tqdm(total=len(wes)) as pbar:  # Crea una barra de progreso
        pool = multiprocessing.Pool(processes=num_processes)
        results = [pool.apply_async(simulate_we, args=(we, C, N, dt, Tmaxneuronal)) for we in wes]

        for result in results:
            maxRateNoFIC.append(result.get())
            pbar.update(1)  # Actualiza la barra de progreso

    ee, = plt.plot(wes, maxRateNoFIC)
    ee.set_label("E-E")

    # Termino los procesos
    pool.close()
    pool.join()

    print("======================================")
    print("=    simulating FIC                  =")
    print("======================================")

    # Resto del código para la simulación FIC
    maxRateFIC = np.zeros(len(wes))
    if precompute:
        BalanceFIC.Balance_AllJ9(C, wes, baseName=fileName)
    for kk, we in enumerate(wes):
        print("\nProcessing: {}  ".format(we), end='')
        DMF.setParms({'we': we})
        balancedJ = BalanceFIC.Balance_J9(we, C, fileName.format(np.round(we, decimals=2)))['J'].flatten()
        integrator.neuronalModel.setParms({'J': balancedJ})
        integrator.recompileSignatures()
        v = integrator.simulate(dt, Tmaxneuronal)[:,1,:]
        maxRateFIC[kk] = np.max(np.mean(v,0))
        print("maxRateFIC => {}".format(maxRateFIC[kk]))
    fic, = plt.plot(wes, maxRateFIC)
    fic.set_label("FIC")

    for line, color in zip([1.47, 4.45], ['r','b']):
        plt.axvline(x=line, label='line at x = {}'.format(line), c=color)
    plt.title("Large-scale network (DMF)" + extraTitle)
    plt.ylabel("Maximum rate (Hz)")
    plt.xlabel("Global Coupling (G = we)")
    plt.legend()
    plt.show()

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================
if __name__ == '__main__':
    plt.rcParams.update({'font.size': 15})

    # Simple verification test, to check the info from the paper...
    print(f"Simple test for verification: phie={DMF.phie(-0.026+DMF.be/DMF.ae)}")
    print("Should print result: phie 3.06308542427")

    # print("Running single node...")
    # N = 1
    # DMF.we = 0.
    # C = np.zeros((N,N))  # redundant, I know...
    # DMF.J = np.ones(N)
    # runAndPlotSim(C, "Single node simulation")

    # Load connectome:
    # --------------------------------
    inFilePath = '../../Data_Raw'
    outFilePath = '../../Data_Produced'
    CFile = sio.loadmat(inFilePath + '/Human_66.mat')  # load Human_66.mat C
    C = CFile['C']
    fileName = outFilePath + '/Human_66/Benji_Human66_{}.mat'  # integrationMode+'Benji_Human66_{}.mat'

    # ================================================================
    # This plots the graphs at Fig 2c of [D*2014]
    plotMaxFrecForAllWe(C, fileName=fileName)

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF
