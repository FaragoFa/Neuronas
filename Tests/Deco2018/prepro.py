# ==========================================================================
# ==========================================================================
#  Computes the Functional Connectivity Dynamics (FCD)
#
#  From the original code:
# --------------------------------------------------------------------------
#  OPTIMIZATION GAIN
#
#  Taken from the code (fgain_Neuro.m) from:
#
#  [DecoEtAl_2018] Deco,G., Cruzat,J., Cabral, J., Knudsen,G.M., Carhart-Harris,R.L., Whybrow,P.C., Logothetis,N.K. & Kringelbach,M.L.
#       Whole-brain multimodal neuroimaging model using serotonin receptor maps explain non-linear functional effects of LSD
#       (2018) Current Biology
#       https://www.cell.com/current-biology/fulltext/S0960-9822(18)31045-5
#
#  Translated to Python & refactoring by Gustavo Patow
# ==========================================================================
# ==========================================================================


# --------------------------------------------------------------------------
#  Begin setup...
# --------------------------------------------------------------------------
from Tests.Deco2018.setup import *
# --------------------------------------------------------------------------
#  End setup...
# --------------------------------------------------------------------------


# ==========================================================================
# ==========================================================================
# ==========================================================================
# IMPORTANT: This function was created to reproduce Deco et al.'s 2018 code for Figure 3A.
# Actually, this only performs the fitting which gives the value of we (we in the original
# code, G in the paper) to use for further computations (e.g., plotting Figure 3A).
# For the plotting, see the respective file (plot.py)
def prepro_G_Optim(fic = None, neuronalModel = None, G_optim = None, M_e_optim = None, M_i_optim = None, J_fileNames = None, distanceSettings = {'FC': (FC, False), 'swFCD': (swFCD, True), 'phFCD': (phFCD, True)}):


    scheme.neuronalModel = neuronalModel
    integrator.neuronalModel = neuronalModel


    tc_transf, C, NumSubjects = init(neuronalModel)
    BalanceFIC.balancingMechanism = fic
    # %%%%%%%%%%%%%%% Set General Model Parameters
    if G_optim is not None:
        neuronalModel.setParms({'G': G_optim})
    if M_e_optim is not None:
        neuronalModel.setParms({'M_e': M_e_optim})
    if M_i_optim is not None:
        neuronalModel.setParms({'M_i': M_i_optim})

    wStart = 0
    Step = 0.1  # 0.025
    wEnd = 10 +0.001
    WEs = np.arange(wStart, wEnd, Step)  # 100 values values for constant G. Originally was np.arange(0,2.5,0.025)


    MStart = 0.5
    MStep = 0.1
    MEnd = 2 + 0.001
    Ms = np.arange(MStart, MEnd, MStep)  # 100 values values for constant G. Originally was np.arange(0,2.5,0.025)

    # Integration parms...
    dt = 0.1
    tmax = 10000.
    Tmaxneuronal = int((tmax + dt))

    # Model Simulations
    # ------------------------------------------
    if fic is not None:
        balancedParms = BalanceFIC.Balance_AllJ9(C, WEs, baseName=J_fileNames)
        modelParms = [balancedParms[i] for i in balancedParms]

    else:
        result = {}
        # for we in WEs:  # iterate over the weight range (G in the paper, we here)
        #     print("\nProcessing: {}  ".format(we), end='')
        #     neuronalModel.setParms({'we': we})
        #     integrator.recompileSignatures()
        #     bestJ = integrator.simulate(dt, Tmaxneuronal)[:, 2, :]
        #     result[we] = {'we': we, 'J': bestJ}


        # if not parallel:
        # for we in WEs:  # iterate over the weight range (G in the paper, we here)
        #     result[we] = {'we': we}
        for M_e in Ms:  # iterate over the weight range (G in the paper, we here)
            result[M_e] = {'M_e': M_e}
        modelParms = [result[i] for i in result]


    # Now, optimize all we (G) values: determine optimal G to work with
    print("\n\n###################################################################")
    print("# Compute G_Optim")
    print("###################################################################\n")
    fitting = optim1D.distanceForAll_Parms(tc_transf, Ms, modelParms,
                                           observablesToUse=distanceSettings, NumSimSubjects=NumSubjects,
                                           parmLabel='M_e',
                                           outFilePath=outFilePath, fileNameSuffix='')

    optimal = {ds: distanceSettings[ds][0].findMinMax(fitting[ds]) for ds in distanceSettings}
    print("Optimal:\n", optimal)

    filePath = outFilePath + 'DecoEtAl2018_fneuro.mat'
    sio.savemat(filePath, #{'JI': JI})
                {'M_e': Ms,
                 'FC_fitt': fitting['FC'],
                 'swFCD_fitt': fitting['swFCD'],
                 'phFCD_fitt': fitting['phFCD'],
                })  # save('fneuro.mat','WE','fitting2','fitting5','FCDfitt2','FCDfitt5');
    print(f"DONE!!! (file: {filePath})")



def save_J (G_optim = None, M_e_optim = None, M_i_optim = None):

    import WholeBrain.Models.Naskar as Naskar
    scheme.neuronalModel = Naskar
    integrator.neuronalModel = Naskar
    tc_transf, C, NumSubjects = init(Naskar)

    if G_optim is not None:
        Naskar.setParms({'G': G_optim})
    if M_e_optim is not None:
        Naskar.setParms({'M_e': M_e_optim})
    if M_i_optim is not None:
        Naskar.setParms({'M_i': M_i_optim})

    dt = 0.1
    tmax = 10000
    Tmaxneuronal = int((tmax + dt))

    print("Saving J...")
    filePath = outFilePath + 'J_Optim.mat'
    J = integrator.warmUpAndSimulate(dt, Tmaxneuronal, TWarmUp=60 * 1000)[tmax-1, 2, :]
    sio.savemat(filePath, {'J': J})
    print(f"DONE!!! (file: {filePath})")

if __name__ == '__main__':
    prepro_G_Optim()

# ==========================================================================
# ==========================================================================
# ==========================================================================EOF
