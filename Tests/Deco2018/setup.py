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
import scipy.io as sio
import os
import csv
import random
import h5py

# --------------------------------------------------------------------------
#  Begin modules setup...
# --------------------------------------------------------------------------
# Setup for Serotonin 2A-based DMF simulation!!!
# This is a wrapper for the DMF (calls it internally, but before switches the
# two gain functions phie and phii for the right ones...


# ----------------------------------------------
neuronalModel = None
import WholeBrain.Integrators.EulerMaruyama as scheme
scheme.sigma = 0.001
import WholeBrain.Integrators.Integrator as integrator
integrator.integrationScheme = scheme
integrator.verbose = False
import WholeBrain.Utils.BOLD.BOLDHemModel_Stephan2008 as Stephan2008
import WholeBrain.Utils.simulate_SimAndBOLD as simulateBOLD
simulateBOLD.integrator = integrator
simulateBOLD.BOLDModel = Stephan2008

# --------------------------------------------------------------------------
# Import optimizer (ParmSweep)
import WholeBrain.Optimizers.ParmSweep as optim1D
optim1D.simulateBOLD = simulateBOLD
optim1D.integrator = integrator

# --------------------------------------------------------------------------
# FIC mechanism
import WholeBrain.Utils.FIC.BalanceFIC as BalanceFIC
BalanceFIC.integrator = integrator
BalanceFIC.balancingMechanism = None

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
import WholeBrain.Observables.phFCD as phFCD

# Escalado de matriz C
scale = 0.1
# --------------------------------------------------------------------------
#  End modules setup...
# --------------------------------------------------------------------------



# --------------------------------------------------------------------------
# File loading…
# --------------------------------------------------------------------------
inFilePath = 'Datos/Datasets/DataHCP80/'
outFilePath = 'Datos/Results/Results_G/'

# outFilePath = 'Datos/Results/Results_test2/'
# inFilePath = 'Datos/Datasets'

fMRI_path = inFilePath + 'hcp1003_{}_LR_dbs80.mat'
SC_path = inFilePath + 'SC_dbs80HARDIFULL.mat'

tasks = ['EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR', 'RELATIONAL', 'REST', 'SOCIAL', 'WM']

# --------------------------------------------------------------------------
# functions to select which subjects to process
# --------------------------------------------------------------------------
# ---------------- load a previously saved list
def loadSubjectList(path):
    subjects = []
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            subjects.append(int(row[0]))
    return subjects


# ---------------- save a freshly created list
def saveSelectedSubjcets(path, subj):
    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for s in subj:
            writer.writerow([s])


# ---------------- fix subset of subjects to sample
def selectSubjects(selectedSubjectsF, maxSubj, numSampleSubj, excluded, forceRecompute=False):
    if not os.path.isfile(selectedSubjectsF) or forceRecompute:  # if we did not already select a list...
        listIDs = random.sample(range(0, maxSubj), numSampleSubj)
        while excluded & set(listIDs):
            listIDs = random.sample(range(0, maxSubj), numSampleSubj)
        saveSelectedSubjcets(selectedSubjectsF, listIDs)
    else:  # if we did, load it!
        listIDs = loadSubjectList(selectedSubjectsF)
    # ---------------- OK, let's proceed
    return listIDs


# --------------------------------------------------------------------------
# functions to load fMRI data for certain subjects
# --------------------------------------------------------------------------
def read_matlab_h5py(filename, excluded):
    with h5py.File(filename, "r") as f:
        # Print all root level object names (aka keys)
        # these can be group or dataset names
        # print("Keys: %s" % f.keys())
        # get first object name/key; may or may NOT be a group
        # a_group_key = list(f.keys())[0]
        # get the object type for a_group_key: usually group or dataset
        # print(type(f['subjects_idxs']))
        # If a_group_key is a dataset name,
        # this gets the dataset values and returns as a list
        # data = list(f[a_group_key])
        # preferred methods to get dataset values:
        # ds_obj = f[a_group_key]  # returns as a h5py dataset object
        # ds_arr = f[a_group_key][()]  # returns as a numpy array

        all_fMRI = {}
        subjects = list(f['subject'])
        for pos, subj in enumerate(subjects):
            print(f'reading subject {pos}')
            group = f[subj[0]]
            try:
                dbs80ts = np.array(group['dbs80ts'])
                all_fMRI[pos] = dbs80ts.T
            except:
                print(f'ignoring register {subj} at {pos}')
                excluded.add(pos)

    return all_fMRI, excluded


def testSubjectData(fMRI_path, excluded):
    print(f'testing {fMRI_path}')
    fMRIs, excluded = read_matlab_h5py(fMRI_path, excluded)  # now, we re only interested in the excluded list
    return excluded


def loadSubjectsData(fMRI_path, listIDs):
    print(f'Loading {fMRI_path}')
    fMRIs, excluded = read_matlab_h5py(fMRI_path, set())   # ignore the excluded list
    res = {}  # np.zeros((numSampleSubj, nNodes, Tmax))
    for pos, s in enumerate(listIDs):
        res[pos] = fMRIs[s]
    return res

# ==================================================================================
#  some useful WholeBrain
# ==================================================================================
@jit(nopython=True)
def initRandom():
    np.random.seed(3)  # originally set to 13


def recompileSignatures(neuronalModel):
    # Recompile all existing signatures. Since compiling isn’t cheap, handle with care...
    # However, this is "infinitely" cheaper than all the other computations we make around here ;-)
    print("\n\nRecompiling signatures!!!")
    neuronalModel.recompileSignatures()
    integrator.recompileSignatures()

def LR_version_symm(TC):
    # returns a symmetrical LR version of AAL 90x90 matrix
    odd = np.arange(0,25,2)
    even = np.arange(1,25,2)[::-1]  # sort 'descend'
    symLR = np.zeros((25,TC.shape[1]))
    symLR[0:13,:] = TC[odd,:]
    symLR[13:25,:] = TC[even,:]
    return symLR


def transformEmpiricalSubjects(tc_aal, NumSubjects):
    transformed = {}
    for s in range(NumSubjects):
        # transformed[s] = np.zeros(tc_aal[0,cond].shape)
        transformed[s] = LR_version_symm(tc_aal[:,:,s])
    return transformed

# ==================================================================================
# ==================================================================================
#  initialization
# ==================================================================================
# ==================================================================================

def init(selectedTask = 'REST', numSampleSubjects = 10):

    initRandom()
    maxSubjects = 1003
    N = 80  # we are using the dbs80 format!

    selectedSubjectsFile = outFilePath + f'selected_{numSampleSubjects}.txt'
    timeseries = {}
    excluded = set()

    if not os.path.isfile(selectedSubjectsFile):
        for task in tasks:
            print(f'----------- Checking: {task} --------------')
            fMRI_task_path = fMRI_path.format(task)
            excluded = testSubjectData(fMRI_task_path, excluded)

    listSelectedIDs = selectSubjects(selectedSubjectsFile, maxSubjects, numSampleSubjects, excluded)

    for task in tasks:
        print(f'----------- Processing: {task} --------------')
        fMRI_task_path = fMRI_path.format(task)
        timeseries[task] = loadSubjectsData(fMRI_task_path, listSelectedIDs)

    # all_fMRI = {s: d for s,d in enumerate(timeseries)}
    # numSubj, nNodes, Tmax = timeseries.shape  # actually, 80, 1200


    tc_transf = timeseries[selectedTask]

    # ------------ Load Structural Connectivity Matrix
    print(f"Loading {SC_path}")
    sc80 = sio.loadmat(SC_path)['SC_dbs80FULL']
    C = sc80 / np.max(sc80) * scale  # Normalization...

    print('Done!!!')

    return tc_transf, C, numSampleSubjects


# ==========================================================================
# ==========================================================================
# ==========================================================================EOF
