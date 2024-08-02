import argparse
import subprocess
import sys
import os
import time
import glob
import gc

import h5py
import numpy as np

from neuronumba.tools.filters import BandPassFilter
from neuronumba.observables.measures import KolmogorovSmirnovStatistic
from neuronumba.observables.ph_fcd import PhFCD
from neuronumba.bold import BoldStephan2008
from neuronumba.simulator.connectivity import Connectivity
from neuronumba.simulator.history import HistoryNoDelays
from neuronumba.simulator.integrators import EulerDeterministic
from neuronumba.simulator.integrators.euler import EulerStochastic
from neuronumba.simulator.models import Naskar2021
from neuronumba.simulator.monitors import RawSubSample
from neuronumba.simulator.simulator import Simulator
from neuronumba.observables.accumulators import ConcatenatingAccumulator
from neuronumba.tools import hdf

tasks = ['EMOTION', 'GAMBLING', 'LANGUAGE', 'MOTOR', 'RELATIONAL', 'REST', 'SOCIAL', 'WM']

subjects_excluded = {515, 330, 778, 140, 77, 74, 335, 591, 978, 596, 406, 729, 282, 667, 157, 224, 290, 355, 930, 742,
                     425, 170, 299, 301, 557, 239, 240, 238, 820, 502, 185, 700}


def read_matlab_h5py(filename, max_subjects=None):
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
            if max_subjects and pos >= max_subjects:
                break

    return all_fMRI


def read_matlab_h5py_single(filename, nsubject):
    with h5py.File(filename, "r") as f:
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


def loadSubjectsData(fMRI_path, max_subjects=None):
    print(f'Loading {fMRI_path}')
    fMRIs = read_matlab_h5py(fMRI_path, max_subjects)  # ignore the excluded list
    return fMRIs


def compute_simulated(signal, period):
    b = BoldStephan2008()
    bold = b.compute_bold(signal, period)
    bpf = BandPassFilter(k=2, flp=0.01, fhi=0.1, tr=2.0)
    bold_filt = bpf.filter(bold.T)
    ph_fcd = PhFCD()
    simulated = ph_fcd.from_fmri(bold_filt)
    return simulated


def process_empirical_subjects(bold_signals, observable, accumulator, bpf):
    # Process the BOLD signals
    # BOLDSignals is a dictionary of subjects {subjectName: subjectBOLDSignal}
    # observablesToUse is a dictionary of {observableName: observablePythonModule}
    ds = 'phFCD'
    num_subjects = len(bold_signals)
    n_rois = bold_signals[next(iter(bold_signals))].shape[
        0]  # get the first key to retrieve the value of N = number of areas

    # First, let's create a data structure for the observables operations...
    measureValues = {}
    measureValues[ds] = accumulator.init(num_subjects, n_rois)

    # Loop over subjects
    for pos, s in enumerate(bold_signals):
        print(
            '   Processing signal {}/{} Subject: {} ({}x{})'.format(pos + 1, num_subjects, s, bold_signals[s].shape[0],
                                                                    bold_signals[s].shape[1]), flush=True)
        signal = bold_signals[s]  # LR_version_symm(tc[s])

        signal_filt = bpf.filter(signal)
        procSignal = observable.from_fmri(signal_filt)
        measureValues[ds] = accumulator.accumulate(measureValues[ds], pos, procSignal[ds])

    measureValues[ds] = accumulator.postprocess(measureValues[ds])

    return measureValues


def prepro():
    parser = argparse.ArgumentParser()

    parser.add_argument("--fitting", help="Start the fitting process", action=argparse.BooleanOptionalAction)
    parser.add_argument("--multi", help="Use multiple fNeuro_emp files", action=argparse.BooleanOptionalAction)
    parser.add_argument("--max-emp_subjects", help="Maximum empirical subjects to use", type=int, required=False)
    parser.add_argument("--post-g-optim", help="Post process G optimization", action=argparse.BooleanOptionalAction)
    parser.add_argument("--post-fitting", help="Post process G optimization", action=argparse.BooleanOptionalAction)
    parser.add_argument("--process-empirical", help="Process empirical subjects", action=argparse.BooleanOptionalAction)
    parser.add_argument("--we", help="G value to explore", type=float, required=False)
    parser.add_argument("--we-range", nargs=3, help="Parameter sweep range for G", type=float, required=False)
    parser.add_argument("--subject", help="Subject to explore", type=int, required=False)
    parser.add_argument("--me-range", nargs=3, help="Parameter sweep range for Me", type=float, required=False)
    parser.add_argument("--mi-range", nargs=3, help="Parameter sweep range for Mi", type=float, required=False)
    parser.add_argument("--task", help="Task to compute", type=str, required=False)

    args = parser.parse_args()

    dir_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(dir_path, '..', '..', 'Datos', 'Datasets', 'DataHCP80')
    out_path = os.path.join(dir_path, '..', '..', 'Datos', 'Results', 'Results_cluster', 'neuronumba')
    fMRI_path = os.path.join(data_path, 'hcp1003_{}_LR_dbs80.mat')
    emp_filename_pattern = os.path.join(out_path, 'fNeuro_emp_{}.mat')

    sc_scale = 0.2

    # Process BOLD sihgnals from subjects obrtained with fMRI, and generate the corresponding observable
    if args.process_empirical:
        for task in tasks:
            emp_filename = emp_filename_pattern.format(task)
            if os.path.exists(emp_filename):
                print(f"File <{emp_filename}> already exists!")
                continue

            bold_signals = loadSubjectsData(fMRI_path.format(task))

            bpf = BandPassFilter(k=2, flp=0.01, fhi=0.1, tr=2.0, apply_detrend=True, apply_demean=True)
            processed = process_empirical_subjects(bold_signals, PhFCD(), ConcatenatingAccumulator(), bpf)
            hdf.savemat(emp_filename, processed)

            print(f'Processed empirical subjects, n = {len(bold_signals)}')

        exit(0)

    # Show distances for all G files generated
    if args.post_g_optim:
        file_list = glob.glob(os.path.join(out_path, 'fitting_g_*.mat'))
        for f in sorted(file_list):
            distance = hdf.loadmat(f)['dist']
            print(f"Distance for <{os.path.basename(f)} = {distance}", flush=True)

        exit(0)

    if args.we:
        out_file = os.path.join(out_path, f'fitting_g_{args.we}.mat')

        if os.path.exists(out_file):
            exit(0)

        sc_filename = 'SC_dbs80HARDIFULL.mat'
        sc_path = os.path.join(data_path, sc_filename)

        fMRI_path = os.path.join(data_path, 'hcp1003_{}_LR_dbs80.mat')

        print(f"Loading {sc_path}")
        sc80 = hdf.loadmat(sc_path)['SC_dbs80FULL']
        C = sc80 / np.max(sc80) * sc_scale  # Normalization...

        n_rois = C.shape[0]
        lengths = np.random.rand(n_rois, n_rois) * 10.0 + 1.0
        speed = 1.0
        dt = 0.1

        processed = hdf.loadmat(emp_filename_pattern.format('REST'))['phFCD']

        accumulator = ConcatenatingAccumulator()
        measure_values = accumulator.init(np.r_[0], n_rois)
        num_sim_subjects = 100

        print(f'Computing distance for G={args.we}', flush=True)
        for n in range(num_sim_subjects):
            history = HistoryNoDelays(weights=C)
            con = Connectivity(weights=C, lengths=lengths, speed=speed)
            m = Naskar2021(g=args.we)
            integ = EulerStochastic(dt=dt, sigmas=np.r_[1e-3, 1e-3, 0])
            # integ = EulerDeterministic(dt=dt)
            monitor = RawSubSample(period=1.0, state_vars=m.get_state_sub(), obs_vars=m.get_observed_sub(['re']))
            s = Simulator(connectivity=con, model=m, history=history, integrator=integ, monitors=[monitor])
            s.configure()
            start_time = time.perf_counter()
            s.run(0, 440000)

            signal = monitor.data('re')
            simulated = compute_simulated(signal, monitor.period)
            measure_values = accumulator.accumulate(measure_values, n, simulated['phFCD'])
            t_dist = time.perf_counter() - start_time
            print(f"Simulated subject {n+1}/{num_sim_subjects} in {t_dist} seconds", flush=True)

        measure = KolmogorovSmirnovStatistic()
        dist = measure.distance(processed, measure_values)
        hdf.savemat(out_file, {'phFCD': measure_values, 'dist': dist})
        exit(0)

    if args.we_range:
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
        exit(0)

    if args.subject is not None:
        optim_g = 3.0
        out_path = os.path.join(out_path, f'subj_{args.subject}')
        os.makedirs(out_path, exist_ok=True)

        sc_filename = 'SC_dbs80HARDIFULL.mat'
        sc_path = os.path.join(data_path, sc_filename)
        print(f"Loading {sc_path}")
        sc80 = hdf.loadmat(sc_path)['SC_dbs80FULL']
        C = sc80 / np.max(sc80) * sc_scale  # Normalization...

        n_rois = C.shape[0]
        lengths = np.random.rand(n_rois, n_rois) * 10.0 + 1.0
        speed = 1.0
        dt = 0.1

        mes = np.arange(args.me_range[0], args.me_range[1], args.me_range[2])
        mis = np.arange(args.mi_range[0], args.mi_range[1], args.mi_range[2])

        task = args.task
        print(f"Starting fitting process for subject {args.subject} for task {task}, {len(mes)} points for M_e and {len(mis)} points for M_i")
        outEmpFileName = out_path + '/fNeuro_emp_' + task + '.mat'
        fMRI = read_matlab_h5py_single(fMRI_path.format(task), args.subject)
        
        bpf = BandPassFilter(k=2, flp=0.01, fhi=0.1, tr=2.0, apply_detrend=True, apply_demean=True)
        processed = process_empirical_subjects({0: fMRI}, PhFCD(), ConcatenatingAccumulator(), bpf)

        start_time = time.perf_counter()

        for me in mes:
            for mi in mis:
                accumulator = ConcatenatingAccumulator()
                measure_values = accumulator.init(np.r_[0], n_rois)
                num_sim_subjects = 10

                print(f'Computing distance for G={optim_g}', flush=True)
                for n in range(num_sim_subjects):
                    history = HistoryNoDelays(weights=C)
                    con = Connectivity(weights=C, lengths=lengths, speed=speed)
                    m = Naskar2021(g=optim_g, M_e=me, M_i=mi)
                    integ = EulerStochastic(dt=dt, sigmas=np.r_[1e-2, 1e-2, 0])
                    monitor = RawSubSample(period=1.0, state_vars=m.get_state_sub(), obs_vars=m.get_observed_sub(['re']))
                    s = Simulator(connectivity=con, model=m, history=history, integrator=integ, monitors=[monitor])
                    s.configure()
                    start_time = time.perf_counter()
                    s.run(0, 440000)

                    signal = monitor.data('re')
                    simulated = compute_simulated(signal, monitor.period)
                    measure_values = accumulator.accumulate(measure_values, n, simulated['phFCD'])
                    print(f"Simulated subject {n+1}/{num_sim_subjects} in {task}_{np.round(me, decimals=3)}_{np.round(mi, decimals=3)}", flush=True)

                measure = KolmogorovSmirnovStatistic()
                dist = measure.distance(processed['phFCD'], measure_values)
                out_file = os.path.join(out_path, f'fitting_{task}_{np.round(me, decimals=3)}_{np.round(mi, decimals=3)}.mat')
                hdf.savemat(out_file, {'phFCD': measure_values, 'dist': dist, 'me': me, 'mi': mi, 'task': task})
                gc.collect()
        
        print(f"Time for subject {args.subject}, task {task}, sweep {fitting_suffix} = {time.perf_counter() - start_time}")
        print("##DONE##", flush=True)
        exit(0)
                    
    if args.fitting:

        num_subjects = 1003
        chunk_size = 20
        list_subjects = list(set(range(0, num_subjects)) - subjects_excluded)
        packs_subjects = [list_subjects[i:i+chunk_size] for i in range(0, len(list_subjects), chunk_size)]
        srun = ['srun', '-n1', '-N1', '--time=4-00', '--exclusive', '--distribution=block:*:*,NoPack']

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
        exit(0)

    if args.post_fitting:
        dir_list = glob.glob(os.path.join(out_path, 'subj*'))
        for dir in sorted(dir_list):
            print(f"Starting processing of {os.path.basename(dir)}")
            file_list = glob.glob(os.path.join(dir, "fitting*.mat"))
            for f in file_list:
                distance = hdf.loadmat(f)['dist']
                print(f"Distance for <{os.path.basename(f)} = {distance}", flush=True)

        exit(0)





if __name__ == "__main__":
    prepro()
