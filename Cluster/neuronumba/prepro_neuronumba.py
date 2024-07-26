import argparse
import hdf5storage
import subprocess
import sys
import os
import time
import glob

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
        print('   Processing signal {}/{} Subject: {} ({}x{})'.format(pos+1, num_subjects, s, bold_signals[s].shape[0], bold_signals[s].shape[1]), flush=True)
        signal = bold_signals[s]  # LR_version_symm(tc[s])

        bpf = BandPassFilter(k=2, flp=0.01, fhi=0.1, tr=2.0)
        signal_filt = bpf.filter(signal)
        procSignal = observable.from_fmri(signal_filt)
        measureValues[ds] = accumulator.accumulate(measureValues[ds], pos, procSignal)

    measureValues[ds] = accumulator.postprocess(measureValues[ds])

    return measureValues


def prepro():

    parser = argparse.ArgumentParser()
    parser.add_argument("--post-g-optim", help="Find optim G", action=argparse.BooleanOptionalAction)
    parser.add_argument("--process-empirical", help="Process empirical subjects", action=argparse.BooleanOptionalAction)
    parser.add_argument("--we", help="G value to explore", type=float, required=False)
    parser.add_argument("--we-range", nargs=3, help="Parameter sweep range for G", type=float, required=False)

    args = parser.parse_args()

    dir_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(dir_path, '..', '..', 'Datos', 'Datasets', 'DataHCP80')
    out_path = os.path.join(dir_path, '..', '..', 'Datos', 'Results', 'Results_cluster', 'neuronumba')
    fMRI_path = os.path.join(data_path, 'hcp1003_{}_LR_dbs80.mat')
    emp_filename = os.path.join(out_path, 'fNeuro_emp.mat')


    if args.process_empirical:
        if os.path.exists(emp_filename):
            raise FileExistsError(f"File <{emp_filename}> already exists!")

        bold_signals = loadSubjectsData(fMRI_path.format("REST"))
        processed = process_empirical_subjects(bold_signals, PhFCD(), ConcatenatingAccumulator())
        sio.savemat(emp_filename, {'phFCD': processed})
        print(f'Processed empirical subjects, n = {len(bold_signals)}')
        exit(0)

    if not os.path.exists(emp_filename):
        raise FileNotFoundError(f"File {emp_filename} does not exists!")


    if args.post_g_optim:
        file_list = glob.glob(os.path.join(out_path,'fitting_g_*.mat'))
        measure = KolmogorovSmirnovStatistic()
        processed = sio.loadmat(emp_filename)['phFCD']
        for f in file_list:
            simulated = sio.loadmat(f)['phFCD']
            distance = measure.distance(processed['phFCD'], simulated)
            print(f"Distance for <{os.path.basename(f)} = {distance}", flush=True)
            

    if args.we:
        out_file = os.path.join(out_path, f'fitting_g_{args.we}.mat')

        if os.path.exists(out_file):
            exit(0)

        sc_filename = 'SC_dbs80HARDIFULL.mat'
        sc_scale = 0.1
        sc_path = os.path.join(data_path, sc_filename)

        fMRI_path = os.path.join(data_path, 'hcp1003_{}_LR_dbs80.mat')

        print(f"Loading {sc_path}")
        sc80 = sio.loadmat(sc_path)['SC_dbs80FULL']
        C = sc80 / np.max(sc80) * sc_scale  # Normalization...

        n_rois = C.shape[0]
        lengths = np.random.rand(n_rois, n_rois)*10.0 + 1.0
        speed = 1.0
        dt = 0.1

        processed = sio.loadmat(emp_filename)['phFCD']

        accumulator = ConcatenatingAccumulator()
        measure_values = accumulator.init(np.r_[0], n_rois)
        num_sim_subjects = 100

        print('\n\n ====================== Model Simulations ======================\n\n')
        for n in range(num_sim_subjects):
            coupling = CouplingLinearNoDelays(weights=C, g=args.we)
            con = Connectivity(weights=C, lengths=lengths, speed=speed)
            m = Naskar()
            integ = EulerStochastic(dt=dt, sigmas=np.r_[1e-3, 0, 0])
            monitor = RawSubSample(period=1.0, state_vars=[], obs_vars=[1])
            s = Simulator(connectivity=con, model=m, coupling=coupling, integrator=integ, monitors=[monitor])
            s.configure()
            start_time = time.perf_counter()
            s.run(0, 440000)

            b = BoldStephan2008()
            signal = monitor.data_observed()[:, 0, :]
            bold = b.compute_bold(signal, monitor.period)
            bpf = BandPassFilter(k=2, flp=0.01, fhi=0.1, tr=2.0)
            bold_filt = bpf.filter(bold)
            ph_fcd = PhFCD()
            simulated = ph_fcd.from_fmri(bold_filt)
            measure_values = accumulator.accumulate(measure_values, n, simulated)
            t_dist = time.perf_counter() - start_time
            print(f"Simulated subject {n}/{num_sim_subjects} in {t_dist} seconds", flush=True)

        hdf5storage.savemat(out_file, {'phFCD': measure_values})

    elif args.we_range:
        WEs = np.arange(args.we_range[0], args.we_range[1], args.we_range[2])

        srun = ['srun', '-n', '1', '-N', '1', '-c', '1', '--time=9-00']
        # srun = ['srun', '-n1', '--exclusive']

        script = [sys.executable, __file__]

        print('Starting srun sweep', flush=True)
        workers = []
        for we in WEs:
            command = [*srun, *script]
            command.extend(['--we', f'{we:.2f}'])
            workers.append(subprocess.Popen(command))
            print(f"Executing: {command}", flush=True)

        print('Waiting for the sweep to finish', flush=True)
        exit_codes = [p.wait() for p in workers]
        print('Sweep finished!', flush=True)
        print(exit_codes)


if __name__ == "__main__":
    prepro()
