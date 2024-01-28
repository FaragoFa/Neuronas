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
def fitt(x = None, fic = None,J_fileNames = None, Xs = None, distanceSettings= None, C= None, numSampleSubjects= None, tc_transf= None):

    # Integration parms...
    dt = 0.1
    tmax = 10000.
    Tmaxneuronal = int((tmax + dt))

    # Model Simulations
    # ------------------------------------------
    if fic is not None:
        balancedParms = BalanceFIC.Balance_AllJ9(C, Xs, baseName=J_fileNames)
        modelParms = [balancedParms[i] for i in balancedParms]

    else:
        result = {}
        for xi in Xs:  # iterate over the weight range (G in the paper, we here)
            result[xi] = {x: xi}
        modelParms = [result[i] for i in result]

    # Now, optimize all we (G) values: determine optimal G to work with
    print("\n\n###################################################################")
    print(f"# Compute {x}_Optim")
    print("###################################################################\n")
    fitting = optim1D.distanceForAll_Parms(tc_transf, Xs, modelParms,
                                           observablesToUse=distanceSettings, NumSimSubjects=numSampleSubjects,
                                           parmLabel=x,
                                           outFilePath=outFilePath, fileNameSuffix='')

    optimal = {ds: distanceSettings[ds][0].findMinMax(fitting[ds]) for ds in distanceSettings}
    print("Optimal:\n", optimal)

    filePath = outFilePath + 'DecoEtAl2018_fneuro.mat'
    sio.savemat(filePath, #{'JI': JI})
                {x: Xs,
                 'FC_fitt': fitting['FC'],
                 'swFCD_fitt': fitting['swFCD'],
                 'phFCD_fitt': fitting['phFCD'],
                })  # save('fneuro.mat','WE','fitting2','fitting5','FCDfitt2','FCDfitt5');
    print(f"DONE!!! (file: {filePath})")

def prepro_G_Optim(fic = None, neuronalModel = None, G_optim = None, Start= 0.5, Step = 0.2, End = 1.5 + 0.001,
                   M_e_optim = None, M_i_optim = None, J_fileNames = None,
                   distanceSettings = {'FC': (FC, False), 'swFCD': (swFCD, True), 'phFCD': (phFCD, True)}):


    scheme.neuronalModel = neuronalModel
    integrator.neuronalModel = neuronalModel
    Xs = np.arange(Start, End, Step)

    tc_transf, C, numSampleSubjects = init(neuronalModel)
    BalanceFIC.balancingMechanism = fic
    # %%%%%%%%%%%%%%% Set General Model Parameters
    if G_optim is not None:
        neuronalModel.setParms({'G': G_optim})
    else:
        fitt(x= 'G', fic= fic, J_fileNames = J_fileNames, Xs = Xs,
             distanceSettings = distanceSettings,tc_transf= tc_transf, C= C, numSampleSubjects= numSampleSubjects)

    if M_e_optim is not None:
        neuronalModel.setParms({'M_e': M_e_optim})
    else:
        fitt(x='M_e', fic=fic, J_fileNames=J_fileNames, Xs = Xs,
             distanceSettings=distanceSettings, tc_transf=tc_transf, C=C, numSampleSubjects=numSampleSubjects)

    if M_i_optim is not None:
        neuronalModel.setParms({'M_i': M_i_optim})
    else:
        fitt(x='M_i', fic=fic, J_fileNames=J_fileNames, Xs = Xs,
             distanceSettings=distanceSettings, tc_transf=tc_transf, C=C, numSampleSubjects=numSampleSubjects)

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
