# ==========================================================================
# ==========================================================================
#  Computes the Functional Connectivity Dynamics (FCD)
#
#  From the original code:
# --------------------------------------------------------------------------
#
#  Computes simulations with the Dynamic Mean Field Model (DMF) using
#  Feedback Inhibitory Control (FIC) and Regional Drug Receptor Modulation (RDRM):
#
#  - the optimal coupling (we=2.1) for fitting the placebo condition
#  - the optimal neuromodulator gain for fitting the LSD condition (wge=0.2)
#
#  Before this, needs the results computed in
#   - prepro_fgain_Neuro.py
#
#  Taken from the code (Code_Figure3.m) from:
#  [DecoEtAl_2018] Deco,G., Cruzat,J., Cabral, J., Knudsen,G.M., Carhart-Harris,R.L., Whybrow,P.C.,
#       Whole-brain multimodal neuroimaging model using serotonin receptor maps explain non-linear functional effects of LSD
#       Logothetis,N.K. & Kringelbach,M.L. (2018) Current Biology
#       https://www.cell.com/current-biology/fulltext/S0960-9822(18)31045-5
#
#  Code written by Josephine Cruzat josephine.cruzat@upf.edu
#
#  Translated to Python by Gustavo Patow
# ==========================================================================
# ==========================================================================
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from pathlib import Path

from Tests.Deco2018.setup import *

def plot_goptim(path_mat = None):
    filePath = path_mat

    print('Loading {}'.format(filePath))
    fNeuro = sio.loadmat(filePath)
    Ms = fNeuro['M_e'].flatten()
    FC_fitt = fNeuro['FC_fitt'].flatten()
    swFCD_fitt = fNeuro['swFCD_fitt'].flatten()
    phFCD_fitt = fNeuro['phFCD_fitt'].flatten()

    maxFC = Ms[np.argmax(FC_fitt)]
    minFCD = Ms[np.argmin(swFCD_fitt)]
    minphFCD = Ms[np.argmin(phFCD_fitt)]
    print("\n\n#####################################################")
    print(f"# Max FC({maxFC}) = {np.max(FC_fitt)} \n "
          f" Min FCD({minFCD}) = {np.min(swFCD_fitt)} \n "
          f" Min phFCD({minphFCD}) = {np.min(phFCD_fitt)}"
           )
    print("#####################################################\n\n")

    plt.rcParams.update({'font.size': 15})
    plotswFCD, = plt.plot(Ms, swFCD_fitt)
    plotswFCD.set_label("swFCD")
    plotphFCD, = plt.plot(Ms, phFCD_fitt)
    plotphFCD.set_label("phFCD")
    plotFC, = plt.plot(Ms, FC_fitt)
    plotFC.set_label("FC")
    plt.title("Whole-brain fitting")
    plt.ylabel("Fitting")
    plt.xlabel("Global Coupling (Ms = M_e)")
    plt.legend()
    plt.show()

# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF
