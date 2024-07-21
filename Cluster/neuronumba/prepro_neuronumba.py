import argparse
import subprocess
import sys
import os


import h5py
import numpy as np
import hdf5storage as sio

from neuronumba.bold.filters import BandPassFilter
from neuronumba.observables.measures import KolmogorovSmirnovStatistic
from neuronumba.observables.ph_fcd import PhFCD
from neuronumba.bold import BoldStephan2008
from neuronumba.simulator.connectivity import Connectivity
from neuronumba.simulator.coupling import CouplingLinearNoDelays
from neuronumba.simulator.integrators import EulerDeterministic
from neuronumba.simulator.integrators.euler import EulerStochastic
from neuronumba.simulator.models import Naskar
from neuronumba.simulator.monitors import RawSubSample
from neuronumba.simulator.simulator import Simulator
from neuronumba.observables.accumulators import ConcatenatingAccumulator


# import WholeBrain.Observables.phFCD as phFCD

# import Models.Naskar as Naskar
# import WholeBrain.Integrators.EulerMaruyama as scheme
# import WholeBrain.Integrators.Integrator as integrator
# scheme.neuronalModel = Naskar
# integrator.integrationScheme = scheme
# integrator.neuronalModel = Naskar
# integrator.verbose = False
# import WholeBrain.Utils.BOLD.BOLDHemModel_Stephan2008 as Stephan2007
# import WholeBrain.Utils.simulate_SimAndBOLD as simulateBOLD
# simulateBOLD.integrator = integrator
# simulateBOLD.BOLDModel = Stephan2007
# import WholeBrain.Optimizers.ParmSweep as optim1D
# optim1D.simulateBOLD = simulateBOLD
# optim1D.integrator = integrator
# import WholeBrain.Observables.phFCD as phFCD

# import WholeBrain.Observables.BOLDFilters as filters
# filters.k = 2                             # 2nd order butterworth filter
# filters.flp = .01                         # lowpass frequency of filter
# filters.fhi = .1                          # highpass
# filters.TR = 2.                           # TR

# from WholeBrain.Utils.preprocessSignal import processBOLDSignals, processEmpiricalSubjects
# import WholeBrain.Optimizers.ParmSweep as ParmSeewp
# ParmSeewp.integrator = integrator
# ParmSeewp.simulateBOLD = simulateBOLD

tasks = ['EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR', 'RELATIONAL', 'REST', 'SOCIAL', 'WM']

subjects_excluded = {515, 330, 778, 140, 77, 74, 335, 591, 978, 596, 406, 729, 282, 667, 157, 224, 290, 355, 930, 742, 425, 170, 299, 301, 557, 239, 240, 238, 820, 502, 185, 700}

def read_matlab_h5py(filename):
    with h5py.File(filename, "r") as f:
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

def process_empirical_subjects(bold_signals, observable, accumulator):
       # Process the BOLD signals
    # BOLDSignals is a dictionary of subjects {subjectName: subjectBOLDSignal}
    # observablesToUse is a dictionary of {observableName: observablePythonModule}
    ds = 'phFCD'
    num_subjects = len(bold_signals)
    n_rois = bold_signals[next(iter(bold_signals))].shape[0]  # get the first key to retrieve the value of N = number of areas

    # First, let's create a data structure for the observables operations...
    measureValues = {}
    measureValues[ds] = accumulator.init(num_subjects, n_rois)


    # Loop over subjects
    for pos, s in enumerate(bold_signals):
        print('   Processing signal {}/{} Subject: {} ({}x{})'.format(pos+1, num_subjects, s, bold_signals[s].shape[0], bold_signals[s].shape[1]), end='', flush=True)
        signal = bold_signals[s]  # LR_version_symm(tc[s])

        bpf = BandPassFilter(k=2, flp=0.01, fhi=0.1, tr=2.0)
        signal_filt = bpf.filter(signal)
        procSignal = observable.from_fmri(signal_filt)
        measureValues[ds] = accumulator.accumulate(measureValues[ds], pos, procSignal)

    measureValues[ds] = accumulator.postprocess(measureValues[ds])

    return measureValues


def prepro():

    parser = argparse.ArgumentParser()
    parser.add_argument("--we", help="G value to explore", type=float, required=False)
    parser.add_argument("--we-range", nargs=3, help="Parameter sweep range for G", type=float, required=False)

    args = parser.parse_args()

    dir_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(dir_path, '..', '..', 'Datos', 'Datasets', 'DataHCP80')
    out_path = os.path.join(dir_path, '..', '..', 'Datos', 'Results', 'Results_cluster', 'neuronumba')
    fMRI_path = os.path.join(data_path, 'hcp1003_{}_LR_dbs80.mat')
    emp_filename = os.path.join(out_path, 'fNeuro_emp.mat')

    if os.path.exists(emp_filename):
        print("Empirical subjects already computed")
    else:
        print("Loading data...")
        fMRIs = loadSubjectsData(fMRI_path.format('REST'))
        process_empirical_subjects(fMRIs, PhFCD(), ConcatenatingAccumulator())
        sio.savemat(emp_filename)

    exit(0)

    # if args.we:

    #     sc_filename = 'SC_dbs80HARDIFULL.mat'
    #     sc_scale = 0.1
    #     sc_path = os.path.join(data_path, sc_filename)

    #     fMRI_path = os.path.join(data_path, 'hcp1003_{}_LR_dbs80.mat')

    #     print(f"Loading {sc_path}")
    #     sc80 = sio.loadmat(sc_path)['SC_dbs80FULL']
    #     C = sc80 / np.max(sc80) * sc_scale  # Normalization...

    #     n_rois = C.shape[0]
    #     lengths = np.random.rand(n_rois, n_rois)*10.0 + 1.0
    #     speed = 1.0
    #     con = Connectivity(weights=C, lengths=lengths, speed=speed)
    #     n_rois = con.n_rois
    #     m = Naskar()
    #     dt = 0.1
    #     integ = EulerStochastic(dt=dt, sigmas=np.r_[1e-3, 0, 0])
    #     # coupling = CouplingLinearDense(weights=weights, delays=con.delays, c_vars=np.array([0], dtype=np.int32), n_rois=n_rois)
    #     coupling = CouplingLinearNoDelays(weights=weights, c_vars=np.array([0], dtype=np.int32), n_rois=n_rois, g=1.0)
    #     # mnt = TemporalAverage(period=1.0, dt=dt)
    #     monitor = RawSubSample(period=1.0, dt=dt)
    #     s = Simulator(connectivity=con, model=m, coupling=coupling, integrator=integ, monitors=[monitor])
    #     s.configure()
    #     s.run(0, 100000)













    #     Naskar.setParms({'SC': C})  # Set the model with the SC
    #     Naskar.couplingOp.setParms(C)

    #     fMRIs = loadSubjectsData(fMRI_path.format('REST'))
    #     observablesToUse = {'phFCD': (phFCD, True)}

    #     print("\n\n*************** Starting: optim1D.distanceForAll_modelParams *****************\n\n")
    #     if True:
    #         import WholeBrain.Utils.decorators as deco
    #         deco.verbose = True

    #     NumSubjects = len(fMRIs)
    #     N = fMRIs[next(iter(fMRIs))].shape[0]  # get the first key to retrieve the value of N = number of areas
    #     print('fMRIs({} subjects): each entry has N={} regions'.format(NumSubjects, N))

    #     emp_filename = out_path + '/fNeuro_emp.mat'
    #     processed = processEmpiricalSubjects(fMRIs, observablesToUse, emp_filename)


    #     # Model Simulations
    #     # -----------------
    #     print('\n\n ====================== Model Simulations ======================\n\n')
    #     parmLabel = 'we'
    #     NumSimSubjects = 100
    #     parm = args.we
    #     # ---- Perform the simulation of NumSimSubjects ----
    #     outFileNamePattern = out_path + '/fitting_' + parmLabel + '{}.mat'
    #     simMeasures = ParmSeewp.evaluateForOneParm(parm, {'we': parm}, NumSimSubjects,
    #                                      observablesToUse, parmLabel,
    #                                      outFileNamePattern.format(np.round(parm, decimals=3)))

    #     # ---- and now compute the final FC, FCD, ... distances for this G (we)!!! ----

    #     print("DONE!!!")

    # else:
    #     WEs = np.arange(args.we_range[0], args.we_range[1], args.we_range[2])

    #     srun = ['srun', '-n', '1', '-N', '1', '-c', '1', '--time=1-00']
    #     # srun = ['srun', '-n1', '--exclusive']

    #     script = [sys.executable, __file__]

    #     print('Starting srun sweep', flush=True)
    #     workers = []
    #     for we in WEs:
    #         command = [*srun, *script]
    #         command.extend(['--we', f'{we}'])
    #         workers.append(subprocess.Popen(command))
    #         print(f"Executing: {command}", flush=True)

    #     print('Waiting for the sweep to finish', flush=True)
    #     exit_codes = [p.wait() for p in workers]
    #     print('Sweep finished!', flush=True)
    #     print(exit_codes)

    #     # print(f"#{we}/{len(np.nditer(modelParams))}:", end='', flush=True)
    #     # for ds in observablesToUse:
    #     #     fitting[ds][pos] = observablesToUse[ds][0].distance(simMeasures[ds], processed[ds])
    #     #     print(f" {ds}: {fitting[ds][pos]};", end='', flush=True)
    #     # print("\n")
    #     #
    #     # print(
    #     #     "\n\n#####################################################################################################")
    #     # print(f"# Results (in ({modelParams[0]}, {modelParams[-1]})):")
    #     # for ds in observablesToUse:
    #     #     optimValDist = observablesToUse[ds][0].findMinMax(fitting[ds])
    #     #     parmPos = [a for a in np.nditer(modelParams)][optimValDist[1]]
    #     #     print(f"# Optimal {ds} =     {optimValDist[0]} @ {np.round(parmPos, decimals=3)}")
    #     # print(
    #     #     "#####################################################################################################\n\n")


if __name__ == "__main__":
    prepro()
