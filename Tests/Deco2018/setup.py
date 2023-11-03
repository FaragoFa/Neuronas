# ==========================================================================
# ==========================================================================
#  Setup for the code from the paper
#
#  [DecoEtAl_2018] Deco,G., Cruzat,J., Cabral, J., Knudsen,G.M., Carhart-Harris,R.L., Whybrow,P.C., Logothetis,N.K. & Kringelbach,M.L.
#       Whole-brain multimodal neuroimaging model using serotonin receptor maps explain non-linear functional effects of LSD
#       (2018) Current Biology
#       https://www.cell.com/current-biology/fulltext/S0960-9822(18)31045-5
#
#  Translated to Python & refactoring by Gustavo Patow
# ==========================================================================
# ==========================================================================


import numpy as np
from numba import jit
import os


# --------------------------------------------------------------------------
#  Begin modules setup...
# --------------------------------------------------------------------------
# Setup for Serotonin 2A-based DMF simulation!!!
# This is a wrapper for the DMF (calls it internally, but before switches the
# two gain functions phie and phii for the right ones...
import WholeBrain.Models.DynamicMeanField as DMF
# ----------------------------------------------
import WholeBrain.Integrators.EulerMaruyama as integrator
integrator.neuronalModel = DMF
integrator.verbose = False
import WholeBrain.Utils.BOLD.BOLDHemModel_Stephan2007 as Stephan2007
import WholeBrain.Utils.simulate_SimAndBOLD as simulateBOLD
simulateBOLD.integrator = integrator
simulateBOLD.BOLDModel = Stephan2007

# --------------------------------------------------------------------------
# Import optimizer (ParmSweep)
import WholeBrain.Optimizers.ParmSweep as optim1D
optim1D.simulateBOLD = simulateBOLD
optim1D.integrator = integrator

# --------------------------------------------------------------------------
# FIC mechanism
import WholeBrain.Utils.FIC.BalanceFIC as BalanceFIC
BalanceFIC.integrator = integrator

# --------------------------------------------------------------------------
# Filters and Observables
# --------------------------------------------------------------------------
# set BOLD filter settings
import WholeBrain.Observables.BOLDFilters as filters
filters.k = 2                             # 2nd order butterworth filter
filters.flp = .01                         # lowpass frequency of filter
filters.fhi = .1                          # highpass
filters.TR = 0.72                           # TR

# import observables
import WholeBrain.Observables.FC as FC
import WholeBrain.Observables.swFCD as swFCD

# --------------------------------------------------------------------------
# setp up IDs...
PLACEBO_cond = 4; LSD_cond = 1   # 1=LSD rest, 4=PLACEBO rest -> The original code used [2, 5] because arrays in Matlab start with 1...
# --------------------------------------------------------------------------
#  End modules setup...
# --------------------------------------------------------------------------



# --------------------------------------------------------------------------
# File loading…
# --------------------------------------------------------------------------
inFilePath = '../../Datos/Datasets'
outFilePath = '../../Datos/Results/Results_25x25/'


# ==================================================================================
#  some useful WholeBrain
# ==================================================================================
@jit(nopython=True)
def initRandom():
    np.random.seed(3)  # originally set to 13


def recompileSignatures():
    # Recompile all existing signatures. Since compiling isn’t cheap, handle with care...
    # However, this is "infinitely" cheaper than all the other computations we make around here ;-)
    print("\n\nRecompiling signatures!!!")
    DMF.recompileSignatures()
    integrator.recompileSignatures()


def LR_version_symm(TC):
    # returns a symmetrical LR version of AAL 90x90 matrix
    odd = np.arange(0,25,2)
    even = np.arange(1,25,2)[::-1]  # sort 'descend'
    symLR = np.zeros((25,TC.shape[1]))
    symLR[0:12,:] = TC[odd,:]
    symLR[12:25,:] = TC[even,:]
    return symLR


def transformEmpiricalSubjects(tc_aal, cond, NumSubjects):
    transformed = {}
    for s in range(NumSubjects):
        # transformed[s] = np.zeros(tc_aal[0,cond].shape)
        transformed[s] = LR_version_symm(tc_aal[s,cond])
    return transformed


# ==================================================================================
# ==================================================================================
#  initialization
# ==================================================================================
# ==================================================================================
initRandom()

# Load Structural Connectivity Matrix
print(f"Loading {inFilePath}/StructuralConnectivity/netmats2_25.txt")

# Cargar los datos de los sujetos desde los archivos de texto
datos_sujetos_25 = np.loadtxt(inFilePath+'/StructuralConnectivity/netmats2_25.txt')

# Reshape para crear matriz 3D
matrices_por_sujeto_25 = datos_sujetos_25.reshape((1003, 25, 25))

# Calcular la matriz de conectividad promedio de todos los sujetos
matriz_conectividad_promedio = np.mean(matrices_por_sujeto_25, axis=0)
matriz_conectividad_promedio = matriz_conectividad_promedio/matriz_conectividad_promedio.max()
C = matriz_conectividad_promedio*0.1

DMF.setParms({'SC': C})  # Set the model with the SC

# #load fMRI data
# print(f"Loading {inFilePath}/fMRI/100206.txt")
#
# # Cargar los datos timeseries de los sujetos desde el archivo de texto
# datos_ts = np.loadtxt(inFilePath+'/fMRI/100206.txt')
#
# # Reshape para crear una matriz 2D (4800, 25)
# matriz_ts = datos_ts.reshape((4800, 25))
# matriz_tsT = matriz_ts.T

# Directorio que contiene los archivos de texto
directory = inFilePath+'/fMRI'

NumSubjects = 10  # Number of Subjects in empirical fMRI dataset, originally 20...
N = 25 # Parcelations
Tmax = 4800 # Total time

# Lista para almacenar las matrices ajustadas individualmente
matrices_individuales = []

# Contador para rastrear el número de archivos cargados
archivos_cargados = 0

# Iterar sobre los archivos en el directorio
for archivo in os.listdir(directory):
    if archivo.endswith('.txt'):
        # Cargar los datos del archivo
        datos_ts = np.loadtxt(os.path.join(directory, archivo))
        # Ajustar la forma a 4800x25
        matriz_ajustada = datos_ts.reshape((Tmax, N))
        matriz_ajustada = matriz_ajustada.T
        # Añadir la matriz ajustada a la lista
        matrices_individuales.append(matriz_ajustada)
        archivos_cargados += 1

    if archivos_cargados >= NumSubjects:
        break

# Convertir la lista de matrices en una matriz tridimensional
matriz_tridimensional = np.dstack(matrices_individuales)

print(f'matriz_tridimensional is (25, 4800) and each entry has N={N} regions and Tmax={Tmax}')


print(f"Simulating {NumSubjects} subjects!")

# ====================== By default, we set up the parameters for the DEFAULT mode:
# Sets the wgaine and wgaini to 0, but using the standard protocol...
# We initialize both to 0, so we have Placebo conditions.
DMF.setParms({'S_E':0., 'S_I':0.})
recompileSignatures()

tc_transf= transformEmpiricalSubjects(matriz_tridimensional[:,:,1], 10, NumSubjects)  # PLACEBO
# FCemp_cotsampling_PLA = G_optim.processEmpiricalSubjects(tc_transf_PLA, distanceSettings, "Data_Produced/SC90/fNeuro_emp_PLA.mat")
# FCemp_PLA = FCemp_cotsampling_PLA['FC']; cotsampling_PLA = FCemp_cotsampling_PLA['swFCD'].flatten()


# ==========================================================================
# ==========================================================================
# ==========================================================================EOF
