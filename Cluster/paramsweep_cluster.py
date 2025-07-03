import argparse
import subprocess
import sys
import os
import time
import gc

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
import WholeBrain.Utils.BOLD.BOLDHemModel_Stephan2008 as Stephan2007
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

import WholeBrain.Utils.decorators as deco
deco.verbose = True

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


def read_matlab_h5py_single(filename, nsubject):
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

        if nsubject in subjects_excluded:
            raise Exception(f"Subject {nsubject} is in the excluded list!")
         
        subjects = list(f['subject'])
        try:
            pos = subjects[nsubject]
            subj = f[pos[0]]
            dbs80ts = np.array(subj['dbs80ts'])
            return dbs80ts.T
        except:
            raise Exception(f"Error loading data for subject {nsubject}")


def getNumSubjects(filename):
    with h5py.File(filename, "r") as f:
        return len(list(f['subject']))



def loadSubjectsData(fMRI_path):
    print(f'Loading {fMRI_path}')
    fMRIs = read_matlab_h5py(fMRI_path)   # ignore the excluded list
    return fMRIs


sim_inf = 10000

def computeFittingFileSuffix(modelParms, task):
    fitting_suffix = ""
    for p, pv in modelParms.items():
        if fitting_suffix:
            fitting_suffix += "_"    
        fitting_suffix += f"{p}_{np.round(pv, decimals=3)}"
    return fitting_suffix + "_" + task


@deco.loadOrCompute
def evaluateForOneParm(subject, modelParms, NumSimSubjects, fitting_suffix,
                       observablesToUse):  # observablesToUse is a dictionary of {name: (observable module, apply filters bool)}
    integrator.neuronalModel.setParms(modelParms)
    integrator.neuronalModel.recompileSignatures()  # just in case the integrator.neuronalModel != neuronalModel here... ;-)
    integrator.recompileSignatures()

    simulatedBOLDs = {}
    start_time = time.perf_counter()
    maxReps = 5
    for nsub in range(NumSimSubjects):  # trials. Originally it was 20.
        print(f"   Simulating subject {subject} with {fitting_suffix} -> subject {nsub}/{NumSimSubjects}!!!", flush=True)
        bds = simulateBOLD.simulateSingleSubject().T
        repetitionsCounter = 0
        while np.isnan(bds).any() or (np.abs(bds) > sim_inf).any():  # This is certainly dangerous, we can have an infinite loop... let's hope not! ;-)
            repetitionsCounter += 1
            if repetitionsCounter == maxReps:
                    dist = processBOLDSignals({0: np.ones((10,50))}, observablesToUse)
                    dist[fitting_suffix] = modelParms
                    return dist
            print(f"      REPEATING simulation for subject {subject}: NaN ({np.isnan(bds).any()}) or inf ({np.max(bds)}) found!!! (trial: {repetitionsCounter})", flush=True)
            bds = simulateBOLD.simulateSingleSubject().T
        simulatedBOLDs[nsub] = bds

    dist = processBOLDSignals(simulatedBOLDs, observablesToUse)
    # now, add {label: currValue} to the dist dictionary, so this info is in the saved file (if using the decorator @loadOrCompute)
    dist[fitting_suffix] = modelParms
    return dist


def prepro():

    parser = argparse.ArgumentParser()
    parser.add_argument("--we", help="G value to use", type=float, required=False)
    parser.add_argument("--subject", help="Subject to explore", type=int, required=False)
    parser.add_argument("--me-range", nargs=3, help="Parameter sweep range for Me", type=float, required=False)
    parser.add_argument("--mi-range", nargs=3, help="Parameter sweep range for Mi", type=float, required=False)
    parser.add_argument("--task", help="Task to compute", type=str, required=False)

    args = parser.parse_args()

    if args.subject is not None:
        dir_path = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(dir_path, '..', 'Datos', 'Datasets', 'DataHCP80')
        out_path = os.path.join(dir_path, '..', 'Datos', 'Results', 'Results_cluster', f'subj_{args.subject}')
        os.makedirs(out_path, exist_ok=True)

        sc_filename = 'SC_dbs80HARDIFULL.mat'
        sc_scale = 0.1
        sc_path = os.path.join(data_path, sc_filename)

        fMRI_path = os.path.join(data_path, 'hcp1003_{}_LR_dbs80.mat')

        print(f"Loading {sc_path}")
        sc80 = sio.loadmat(sc_path)['SC_dbs80FULL']
        C = sc80 / np.max(sc80) * sc_scale  # Normalization...

        Naskar.setParms({'SC': C})  # Set the model with the SC
        Naskar.couplingOp.setParms(C)


        mes = np.arange(args.me_range[0], args.me_range[1], args.me_range[2])
        mis = np.arange(args.mi_range[0], args.mi_range[1], args.mi_range[2])

        start_time = time.perf_counter()

        task = args.task
        print(f"Starting fitting process for subject {args.subject} for task {task}, {len(mes)} points for M_e and {len(mis)} points for M_i")
        outEmpFileName = out_path + '/fNeuro_emp_' + task + '.mat'
        fMRI = read_matlab_h5py_single(fMRI_path.format(task), args.subject)
        processed = processEmpiricalSubjects({0: fMRI}, {'phFCD': (phFCD, True)}, outEmpFileName)
        for me in mes:
            for mi in mis:
                observablesToUse = {'phFCD': (phFCD, True)}
                N = fMRI.shape[0]  # get the first key to retrieve the value of N = number of areas
                NumSimSubjects = 10
                params = {'we': args.we, 'M_e': me, 'M_i': mi}
                # ---- Perform the simulation of NumSimSubjects ----
                fitting_suffix = computeFittingFileSuffix(params, task)
                outFileName = out_path + '/fitting_' + fitting_suffix + '.mat'
                simMeasures = evaluateForOneParm(args.subject, params, NumSimSubjects, fitting_suffix,
                                                    observablesToUse,
                                                    outFileName)
                gc.collect()
        
        print(f"Time for subject {args.subject}, task {task}, sweep {fitting_suffix} = {time.perf_counter() - start_time}")
        print("##DONE##", flush=True)
        exit(0)
                    

    else:

        num_subjects = 1003
        chunk_size = 20
        list_subjects = list(set(range(0, num_subjects)) - subjects_excluded)
        packs_subjects = [list_subjects[i:i+chunk_size] for i in range(0, len(list_subjects), chunk_size)]
        srun = ['srun', '-n1', '-N1', '--time=2-00', '--exclusive', '--distribution=block:*:*,NoPack']

        script = [sys.executable, __file__]


        finished = []
        workers = []
        max_proc = 200
        for pack in packs_subjects:
            for task in tasks:
                print(f'Starting srun sweep for task {task} with pack [{pack[0]}:{pack[-1]}]', flush=True)
                for ns in pack:
                    for me in np.arange(0.8, 1.5001, 0.05):
                        command = [*srun, *script]
                        command.extend(['--subject', f'{ns}'])
                        command.extend(['--task', task])
                        command.extend(['--we', '15'])
                        command.extend(['--me-range', f'{me}', f'{me+0.0999}', '0.1'])
                        command.extend(['--mi-range', '0.3', '1.1001', '0.05'])
                        workers.append(subprocess.Popen(command))
                        print(f"Executing: {command}", flush=True)
                        while len(workers) >= max_proc:
                            print(f"Maximum concurrent process {len(workers)} >= {max_proc} reached. Waiting ...", flush=True)
                            new_workers = []
                            for p in workers:
                                try:
                                    p.wait(timeout=1)
                                    finished.append(p)
                                    print(f"Process {p.pid} finished!", flush=True)
                                except:
                                    new_workers.append(p)
                            workers = new_workers

        print('Waiting for the sweep to finish', flush=True)
        exit_codes = [p.wait() for p in workers]
        print(f'Sweep finished for task {task} and pack [{pack[0]}:{pack[-1]}]!', flush=True)
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
