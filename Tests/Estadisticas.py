# ==========================================================================
# ==========================================================================
import numpy as np
import scipy.io as sio
import WholeBrain.Utils.p_values as p_values
import matplotlib.pyplot as plt

def loadStates(inFilePath = 'Datos/Results/'):
    # here load the results of the cohort abd burden, and RETURN them as an array of floats
    REST_path = inFilePath + 'Results_Me_REST/J_Optim'
    J_REST = sio.loadmat(REST_path)['J']

    GAMBLING_path = inFilePath + 'Results_Me_GAMBLING/J_Optim'
    J_GAMBLING = sio.loadmat(GAMBLING_path)['J']

    SOCIAL_path = inFilePath + 'Results_Me_SOCIAL/J_Optim'
    J_SOCIAL = sio.loadmat(SOCIAL_path)['J']

    dataSetLabels = ['J_REST', 'J_GAMBLING', 'J_SOCIAL']  # the set of labels over we will iterate...

    # Juntar todos los estados
    States = {}
    for state in dataSetLabels:
        States[state] = dataSetLabels[state]

    # Asignar datos a cada columna del structured array
    States['J_REST'] = J_REST
    States['J_GAMBLING'] = J_GAMBLING
    States['J_SOCIAL'] = J_SOCIAL

    return States, dataSetLabels

if __name__ == '__main__':
    plt.rcParams.update({'font.size': 15})
    # --------------------------------------------------
    # Some more setups...
    # --------------------------------------------------
    dataSetLabels = []  # the set of labels over we will iterate...

    # --------------------------------------------------
    # Load results simulation
    # --------------------------------------------------
    burden='ABeta'
    resI = {}
    for cohort in dataSetLabels:
        cohort_results = loadResultsCohort(base_folder, cohort=cohort, burden=burden)
        resI[cohort] = cohort_results

    # --------------------------------------------------
    # Plot p_value comparison!
    # --------------------------------------------------
    p_values.plotComparisonAcrossLabels2(resI, columnLables=dataSetLabels, graphLabel=f'Results from Simulation for {burden}')

    print('done!')

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF