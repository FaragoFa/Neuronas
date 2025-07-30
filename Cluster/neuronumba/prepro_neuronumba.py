import argparse
from concurrent.futures import ProcessPoolExecutor
import subprocess
import sys
import os
import time
import glob
import gc
import re
import matplotlib.pyplot as plt
import seaborn as sns

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
                all_fMRI[pos] = dbs80ts
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
            return dbs80ts
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
    bold_filt = bpf.filter(bold)
    ph_fcd = PhFCD()
    simulated = ph_fcd.from_fmri(bold_filt)
    return simulated


def process_empirical_subjects(bold_signals, observable, accumulator, bpf):
    # Process the BOLD signals
    # BOLDSignals is a dictionary of subjects {subjectName: subjectBOLDSignal}
    # observablesToUse is a dictionary of {observableName: observablePythonModule}
    ds = 'phFCD'
    num_subjects = len(bold_signals)
    # get the first key to retrieve the value of N = number of areas
    n_rois = bold_signals[next(iter(bold_signals))].shape[1]  

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

def compute_subject(exec_env):
    C = exec_env['C']
    lengths = exec_env['lengths']
    speed = exec_env['speed']
    dt = exec_env['dt']
    n = exec_env['n']
    we = exec_env['we']
    num_sim_subjects = exec_env['num_sim_subjects']

    history = HistoryNoDelays()
    con = Connectivity(weights=C, lengths=lengths, speed=speed)
    m = Naskar2021(weights=C, g=we).configure()
    integ = EulerStochastic(dt=dt, sigmas=np.r_[1e-3, 1e-3, 0])
    # integ = EulerDeterministic(dt=dt)
    monitor = RawSubSample(period=1.0, monitor_vars=m.get_var_info(['S_e']))
    s = Simulator(connectivity=con, model=m, history=history, integrator=integ, monitors=[monitor]).configure()
    start_time = time.perf_counter()
    s.run(0, 440000)

    signal = monitor.data('S_e')
    simulated = compute_simulated(signal, monitor.period)
    t_dist = time.perf_counter() - start_time
    print(f"Simulated subject {n+1}/{num_sim_subjects} in {t_dist} seconds", flush=True)

    return simulated


def prepro():
    parser = argparse.ArgumentParser()

    dir_path = os.path.dirname(os.path.abspath(__file__))

    parser.add_argument("--fitting-subj", type=int, help="Start the fitting process for an specific subject")
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
    parser.add_argument("--task", help="Task to compute", type=str, required=False, default='REST')
    parser.add_argument("--nproc", type=int, default=32, help="Number of processes to use for parallel processing")
    parser.add_argument("--out-path", type=str, default=None, help="Output path for results", required=True)
    parser.add_argument("--data-path", type=str, help="Input path witj subjects data", required=False, default=os.path.join(dir_path, '..', '..', 'Datos', 'Datasets', 'DataHCP80'))
    parser.add_argument("--sc-scale", type=float, default=0.2, help="Scale for the structural connectivity matrix")
    parser.add_argument("--use-mp", help="Use multiprocessing for the fitting process", action=argparse.BooleanOptionalAction)
    parser.add_argument("--num-subjects", type=int, default=100, help="Number of subjects to simulate for each G value")

    args = parser.parse_args()

    data_path = args.data_path
    out_path = args.out_path
    os.makedirs(out_path, exist_ok=True)
    fMRI_path = os.path.join(data_path, 'hcp1003_{}_LR_dbs80.mat')
    emp_filename_pattern = os.path.join(out_path, 'fNeuro_emp_{}.mat')

    sc_scale = args.sc_scale

    # Process BOLD signals from subjects obrtained with fMRI, and generate the corresponding observable
    if args.process_empirical:
        tasks_to_process = [args.task] if args.task else tasks
        for task in tasks_to_process:
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
        x = []
        y = []
        for f in sorted(file_list):
            m = hdf.loadmat(f)
            we = m['we']
            d = m['dist']
            print(f"Distance for we={we} = {d}", flush=True)
            x.append(we)
            y.append(d)

        fig, ax = plt.subplots()
        ax.plot(x, y)
        plt.savefig("fig_g_optim.png", dpi=300)

        exit(0)

    # Compute distance for a specific G value, using the cluster with slurm
    if args.we and not args.use_mp:
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
        num_sim_subjects = args.num_subjects

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

    # Compute distance for a specific G value and a given task, NOT USING SLURM, but multiprocessing
    if args.we and args.use_mp:
        out_file = os.path.join(out_path, f'fitting_{args.task}_g_{args.we}.mat')

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

        processed = hdf.loadmat(emp_filename_pattern.format(args.task))['phFCD']

        accumulator = ConcatenatingAccumulator()
        measure_values = accumulator.init(np.r_[0], n_rois)
        num_sim_subjects = args.num_subjects

        print(f'Computing distance for task {args.task} and G={args.we}', flush=True)

        subjects = list(range(num_sim_subjects))    
        results = []
        while len(subjects) > 0:
            print(f'Creating process pool with {args.nproc} workers')
            pool = ProcessPoolExecutor(max_workers=args.nproc)
            futures = []
            print(f"EXECUTOR --- START cycle for {len(subjects)} subjects")
            for n in subjects:
                exec_env = {
                    'n': n,
                    'we': args.we,
                    'C': C,
                    'lengths': lengths,
                    'speed': speed,
                    'dt': dt,
                    'num_sim_subjects': num_sim_subjects,
                }
                futures.append((n, pool.submit(compute_subject, exec_env)))

            while any(f.done() for _, f in futures):
                time.sleep(5)

            subjects = []
            for n, f in futures:
                try:
                    result = f.result()
                    results.append((n, result))
                    print(f"EXECUTOR --- FINISHED process for subject {n}")
                except Exception as exc:
                    f.cancel()
                    print(f"EXECUTOR --- FAIL. Restarting process for subject {n}")
                    subjects.append(n)


        for n, r in results:
            measure_values = accumulator.accumulate(measure_values, n, r['phFCD'])

        measure = KolmogorovSmirnovStatistic()
        dist = measure.distance(processed, measure_values)
        print(f"Distance for g={args.we} = {dist}")
        hdf.savemat(out_file, {'phFCD': measure_values, 'dist': dist, 'we': args.we})
        exit(0)

    # Sweep for G values with the cluster using slurm
    if args.we_range:
        WEs = np.arange(args.we_range[0], args.we_range[1], args.we_range[2])

        srun = ['srun', '-n', '1', '-N', '1', '-c', '1', '--time=9-00']
        # srun = ['srun', '-n1', '--exclusive']

        script = [sys.executable, __file__]

        print('Starting srun sweep', flush=True)
        workers = []
        for we in WEs:
            command = [*srun, *script]
            command.extend(['--we', f'{we:.2f}', '--task', args.task])
            workers.append(subprocess.Popen(command))
            print(f"Executing: {command}", flush=True)

        print('Waiting for the sweep to finish', flush=True)
        exit_codes = [p.wait() for p in workers]
        print('Sweep finished!', flush=True)
        print(exit_codes)
        exit(0)

    # Perfomr the M_e/M_i fitting for a specific subject and for a task
    if args.subject is not None:
        optim_g = 2.05 # This is the G value that was optimized for the Naskar2021 model
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
        fMRI = read_matlab_h5py_single(fMRI_path.format(task), args.subject)
        
        bpf = BandPassFilter(k=2, flp=0.01, fhi=0.1, tr=2.0, apply_detrend=True, apply_demean=True)
        processed = process_empirical_subjects({0: fMRI}, PhFCD(), ConcatenatingAccumulator(), bpf)


        for me in mes:
            for mi in mis:
                start_time = time.perf_counter()
                fitting_suffix = f"{task}_{np.round(me, decimals=3)}_{np.round(mi, decimals=3)}"
                out_file = os.path.join(out_path, f'fitting_{fitting_suffix}.mat')
                if os.path.exists(out_file):
                    print(f"File already exists for {task}_{np.round(me, decimals=3)}_{np.round(mi, decimals=3)}")
                    continue

                accumulator = ConcatenatingAccumulator()
                measure_values = accumulator.init(np.r_[0], n_rois)
                num_sim_subjects = args.num_subjects


                print(f'Computing distance for subject {args.subject} with M_e={me} M_i={mi}', flush=True)
                for n in range(num_sim_subjects):
                    history = HistoryNoDelays()
                    con = Connectivity(weights=C, lengths=lengths, speed=speed)
                    m = Naskar2021(g=optim_g, M_e=me, M_i=mi)
                    integ = EulerStochastic(dt=dt, sigmas=np.r_[1e-3, 1e-3, 0])
                    monitor = RawSubSample(period=1.0, state_vars=m.get_state_sub(), obs_vars=m.get_observed_sub(['re']))
                    s = Simulator(connectivity=con, model=m, history=history, integrator=integ, monitors=[monitor])
                    s.configure()
                    start_time = time.perf_counter()
                    s.run(0, 440000)

                    signal = monitor.data('re')
                    simulated = compute_simulated(signal, monitor.period)
                    measure_values = accumulator.accumulate(measure_values, n, simulated['phFCD'])
                    print(f"For pacient {args.subject}, finished simulation of subject {n+1}/{num_sim_subjects} in {task}_{np.round(me, decimals=3)}_{np.round(mi, decimals=3)}", flush=True)

                measure = KolmogorovSmirnovStatistic()
                dist = measure.distance(processed['phFCD'], measure_values)
                hdf.savemat(out_file, {'phFCD': measure_values, 'dist': dist, 'me': me, 'mi': mi, 'task': task})
                gc.collect()
        
                print(f"Time for subject {args.subject}, point {fitting_suffix} = {time.perf_counter() - start_time}")
        
        exit(0)


    # Perform the M_e/M_i fitting for all subjects, and all tasks using the cluster with slurm
    if args.fitting:
        num_subjects = 1003
        list_subjects = list(set(range(6, num_subjects)) - subjects_excluded)
        srun = ['srun', '-n 1', '-N 1', '-c 1', '--time=4-00', '--exclusive']

        script = [sys.executable, __file__]

        finished = []
        workers = []
        max_proc = 200
        for ns in list_subjects:
            for task in tasks:
                print(f'Starting srun sweep for task {task} for subject {ns}', flush=True)
                for me in np.arange(0.05, 1.5001, 0.05):
                    for mi in np.arange(0.05, 1.5001, 0.05):
                        command = [*srun, *script]
                        command.extend(['--subject', f'{ns}'])
                        command.extend(['--task', task])
                        command.extend(['--me-range', f'{me}', f'{me+0.001}', '0.1'])
                        command.extend(['--mi-range', f'{mi}', f'{mi+0.001}', '0.1'])
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
        print(exit_codes)
        exit(0)

    # Perform the M_e/M_i fitting for a single subject, and all tasks using the cluster with slurm
    if args.fitting_subj is not None:

        ns = args.fitting_subj
        srun = ['srun', '-n 1', '-N 1', '-c 1', '--time=4-00', '--exclusive']

        script = [sys.executable, __file__]

        finished = []
        workers = []
        max_proc = 200
        for task in tasks:
            print(f'Starting srun sweep for task {task} for subject {ns}', flush=True)
            for me in np.arange(0.05, 1.5001, 0.05):
                for mi in np.arange(0.05, 1.5001, 0.05):
                    command = [*srun, *script]
                    command.extend(['--subject', f'{ns}'])
                    command.extend(['--task', task])
                    command.extend(['--me-range', f'{me}', f'{me+0.001}', '0.1'])
                    command.extend(['--mi-range', f'{mi}', f'{mi+0.001}', '0.1'])
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

        exit_codes = [p.wait() for p in workers]
        print(f'Sweep finished for task {task} and subject {ns}!', flush=True)
        print(exit_codes)
        exit(0)

    # Generate the heatmaps for the M_e/M_i fitting results
    if args.post_fitting:
        dir_list = glob.glob(os.path.join(out_path, 'subj*'))
        for dir in sorted(dir_list):
            subj_match = re.search(r"subj_([0-9]+)", dir)
            s = int(subj_match.group(1))
            print(f"Starting processing of subject {s}")
            for task in tasks:
                fit_files = glob.glob(os.path.join(dir, f"fitting*_{task}_*.mat"))

                if len(fit_files) == 0:
                    continue

                numParms = len(fit_files)
                fitting = {}

                mes = set()
                mis = set()
                data = {}

                for pos, file in enumerate(fit_files):
                    simMeasures = hdf.loadmat(file)
                    result = re.search(r"_([0-9]+.[0-9]+)_([0-9]+.[0-9]+)\.", file) 
                    me = result.group(1)
                    mi = result.group(2)
                    mes.add(me)
                    mis.add(mi)

                    d = simMeasures['dist']
                    data[(me, mi)] = d

                x_me = list(mes)
                y_mi = list(mis)
                x_me.sort(key=lambda k : float(k))
                y_mi.sort(key=lambda k : float(k), reverse=True)

                mat = np.zeros((len(x_me), len(y_mi)))

                for k, v in data.items():
                    x_pos = x_me.index(k[0])
                    y_pos = y_mi.index(k[1])
                    mat[x_pos, y_pos] = v

                # fig, ax = plt.subplots()
                # im = ax.imshow(mat)

                # # Show all ticks and label them with the respective list entries
                # ax.set_xticks(np.arange(len(x_me)), labels=x_me)
                # ax.set_yticks(np.arange(len(y_mi)), labels=y_mi)
                # for i in range(len(x_me)):
                #     for j in range(len(y_mi)):
                #         text = ax.text(j, i, np.round(mat[i, j], decimals=3),
                #                        ha="center", va="center", color="w")

                ax = sns.heatmap(mat.T, vmin=0, vmax=1, square=True, cmap='viridis', annot_kws={"size": 4}, xticklabels=x_me, yticklabels=y_mi, annot=True)
                ax.set_title(f"Subject {s}, task {task}")
                plt.xlabel("M_e")
                plt.ylabel("M_i")
                plt.yticks(rotation=0) 

                plt.savefig(os.path.join(dir, f"fit_{s}_{task}.png"), dpi=300)
                plt.clf()



if __name__ == "__main__":
    prepro()
