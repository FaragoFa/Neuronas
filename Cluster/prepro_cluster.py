import argparse
import subprocess
import sys
import os


import h5py
import numpy as np
import hdf5storage as sio
import WholeBrain.Observables.phFCD as phFCD

import Models.Naskar as Naskar
import WholeBrain.Integrators.EulerMaruyama as scheme
import WholeBrain.Integrators.Integrator as integrator
scheme.neuronalModel = Naskar
integrator.integrationScheme = scheme
integrator.neuronalModel = Naskar
integrator.verbose = False
import WholeBrain.Utils.BOLD.BOLDHemModel_Stephan2007 as Stephan2007
import WholeBrain.Utils.simulate_SimAndBOLD as simulateBOLD
simulateBOLD.integrator = integrator
simulateBOLD.BOLDModel = Stephan2007
import WholeBrain.Optimizers.ParmSweep as optim1D
optim1D.simulateBOLD = simulateBOLD
optim1D.integrator = integrator
import WholeBrain.Observables.phFCD as phFCD

import WholeBrain.Observables.BOLDFilters as filters
filters.k = 2                             # 2nd order butterworth filter
filters.flp = .01                         # lowpass frequency of filter
filters.fhi = .1                          # highpass
filters.TR = 2.                           # TR

from WholeBrain.Utils.preprocessSignal import processBOLDSignals, processEmpiricalSubjects
import WholeBrain.Optimizers.ParmSweep as ParmSeewp
ParmSeewp.integrator = integrator
ParmSeewp.simulateBOLD = simulateBOLD

tasks = ['EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR', 'RELATIONAL', 'REST', 'SOCIAL', 'WM']

subjects_excluded = {515, 330, 778, 140, 77, 74, 335, 591, 978, 596, 406, 729, 282, 667, 157, 224, 290, 355, 930, 742, 425, 170, 299, 301, 557, 239, 240, 238, 820, 502, 185, 700}

def read_matlab_h5py(filename):
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
            group = f[subj[0]]
            if pos in subjects_excluded:
                continue
            try:
                dbs80ts = np.array(group['dbs80ts'])
                all_fMRI[pos] = dbs80ts.T
            except:
                print(f'ignoring register {subj} at {pos}')

    return all_fMRI


def loadSubjectsData(fMRI_path):
    print(f'Loading {fMRI_path}')
    fMRIs = read_matlab_h5py(fMRI_path)   # ignore the excluded list
    return fMRIs


def prepro():

    parser = argparse.ArgumentParser()
    parser.add_argument("--we", help="G value to explore", type=float, required=False)
    parser.add_argument("--we-range", nargs=3, help="Parameter sweep range for G", type=float, required=False)

    args = parser.parse_args()

    if args.we:
        dir_path = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(dir_path, '..', 'Datos', 'Datasets', 'DataHCP80')
        out_path = os.path.join(dir_path, '..', 'Datos', 'Results', 'Results_cluster')

        sc_filename = 'SC_dbs80HARDIFULL.mat'
        sc_scale = 0.1
        sc_path = os.path.join(data_path, sc_filename)

        fMRI_path = os.path.join(data_path, 'hcp1003_{}_LR_dbs80.mat')

        print(f"Loading {sc_path}")
        sc80 = sio.loadmat(sc_path)['SC_dbs80FULL']
        C = sc80 / np.max(sc80) * sc_scale  # Normalization...

        Naskar.setParms({'SC': C})  # Set the model with the SC
        Naskar.couplingOp.setParms(C)

        fMRIs = loadSubjectsData(fMRI_path.format('REST'))
        observablesToUse = {'phFCD': (phFCD, True)}

        print("\n\n*************** Starting: optim1D.distanceForAll_modelParams *****************\n\n")
        if True:
            import WholeBrain.Utils.decorators as deco
            deco.verbose = True

        fileNameSuffix = ''
        NumSubjects = len(fMRIs)
        N = fMRIs[next(iter(fMRIs))].shape[0]  # get the first key to retrieve the value of N = number of areas
        print('fMRIs({} subjects): each entry has N={} regions'.format(NumSubjects, N))

        outEmpFileName = out_path + '/fNeuro_emp' + fileNameSuffix + '.mat'
        processed = processEmpiricalSubjects(fMRIs, observablesToUse, outEmpFileName)


        # Model Simulations
        # -----------------
        print('\n\n ====================== Model Simulations ======================\n\n')
        parmLabel = 'we'
        NumSimSubjects = 100
        parm = args.we
        # ---- Perform the simulation of NumSimSubjects ----
        outFileNamePattern = out_path + '/fitting_' + parmLabel + '{}' + fileNameSuffix + '.mat'
        simMeasures = ParmSeewp.evaluateForOneParm(parm, {'we': parm}, NumSimSubjects,
                                         observablesToUse, parmLabel,
                                         outFileNamePattern.format(np.round(parm, decimals=3)))

        # ---- and now compute the final FC, FCD, ... distances for this G (we)!!! ----

        print("DONE!!!")

    else:
        WEs = np.arange(args.we_range[0], args.we_range[1], args.we_range[2])

        srun = ['srun', '-n', '1', '-N', '1', '-c', '1', '--time=1-00']
        # srun = ['srun', '-n1', '--exclusive']

        script = [sys.executable, __file__]

        print('Starting srun sweep', flush=True)
        workers = []
        for we in WEs:
            command = [*srun, *script]
            command.extend(['--we', f'{we}'])
            workers.append(subprocess.Popen(command))
            print(f"Executing: {command}", flush=True)

        print('Waiting for the sweep to finish', flush=True)
        exit_codes = [p.wait() for p in workers]
        print('Sweep finished!', flush=True)
        print(exit_codes)

        # print(f"#{we}/{len(np.nditer(modelParams))}:", end='', flush=True)
        # for ds in observablesToUse:
        #     fitting[ds][pos] = observablesToUse[ds][0].distance(simMeasures[ds], processed[ds])
        #     print(f" {ds}: {fitting[ds][pos]};", end='', flush=True)
        # print("\n")
        #
        # print(
        #     "\n\n#####################################################################################################")
        # print(f"# Results (in ({modelParams[0]}, {modelParams[-1]})):")
        # for ds in observablesToUse:
        #     optimValDist = observablesToUse[ds][0].findMinMax(fitting[ds])
        #     parmPos = [a for a in np.nditer(modelParams)][optimValDist[1]]
        #     print(f"# Optimal {ds} =     {optimValDist[0]} @ {np.round(parmPos, decimals=3)}")
        # print(
        #     "#####################################################################################################\n\n")


if __name__ == "__main__":
    prepro()
