# MonSim:  Monitor and Simulator replication of Particle Photon Application
# Copyright (C) 2026 Dave Gutz
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation;
# version 2.1 of the License.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# See http://www.fsf.org/licensing/licenses/lgpl.txt for full license text.

""" Python model of what's installed on the Particle Photon.  Includes
a monitor object (MON) and a simulation object (SIM).   The monitor is
the EKF and Coulomb Counter.   The SIM is a battery model, that also has a
Coulomb Counter built in."""

from MonSim import replicate, save_clean_file, UserOptions
from unite_pictures import cleanup_fig_files, precleanup_fig_files, pngs_to_pdf
from CompareFault import over_fault
from Util import rename_all, save_struct_to_csv
import matplotlib.pyplot as plt
from datetime import datetime
from load_data import load_data
from DataOverModel import dom_plot
import easygui
import sys
from PlotKiller import show_killer
import tkinter.messagebox
from local_paths import version_from_data_file, local_paths
import os
from pathlib import Path, PurePosixPath
from plot.PlotOptions import PlotOptions
plt.rcParams['axes.grid'] = True
plt.rcParams['legend.fontsize'] = 'small'

# Suppress all UserWarning messages
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np


def shift_time(obj, n_steps, fields=None):
    """Shift fields of a struct-like object by n_steps positions.

    fields: iterable of attribute names to shift. If None (default), shifts the
            time column ('time' or 'cTime' if present), back/forward-extrapolating
            gaps using the local dt — original behavior. If provided, shifts each
            named data field instead; gaps are filled with the boundary value
            (data fields aren't necessarily monotonic so dt extrapolation doesn't
            apply).

    n_steps > 0: shift right (each position gets the value from n_steps earlier rows).
    n_steps < 0: shift left (each position gets the value from n_steps later rows).
    All other columns are unmodified.  Returns obj for chaining.
    """
    if obj is None or n_steps == 0:
        return obj

    if fields is None:
        t_col = None
        for candidate in ('time', 'cTime'):
            if hasattr(obj, candidate):
                raw = getattr(obj, candidate)
                if raw is not None and hasattr(raw, '__len__') and len(raw) > 1:
                    t_col = candidate
                    break

        if t_col is None:
            return obj

        time = np.asarray(getattr(obj, t_col), dtype=float)
        n = len(time)
        shifted = np.roll(time, n_steps)

        if n_steps > 0:
            # gap at start: extrapolate backward from time[0] using leading dt
            dt = time[1] - time[0]
            shifted[:n_steps] = time[0] - np.arange(n_steps, 0, -1) * dt
        else:
            # gap at end: extrapolate forward from time[-1] using trailing dt
            dt = time[-1] - time[-2]
            shifted[n + n_steps:] = time[-1] + np.arange(1, -n_steps + 1) * dt

        setattr(obj, t_col, shifted)
        return obj

    for field in fields:
        if not hasattr(obj, field):
            continue
        raw = getattr(obj, field)
        if raw is None or not hasattr(raw, '__len__') or len(raw) <= abs(n_steps):
            continue
        arr = np.asarray(raw, dtype=float)
        n = len(arr)
        shifted = np.roll(arr, n_steps)
        if n_steps > 0:
            shifted[:n_steps] = arr[0]
        else:
            shifted[n + n_steps:] = arr[-1]
        setattr(obj, field, shifted)

    return obj


# noinspection PyPep8Naming
def compare_run_sim(data_file=None, unit_key=None, time_end=None, plots=True, Dw=0.,  use_mon_soc_=False,
                    verbose=False, scale_batt=1., slr_hys_sim=1., request_history=5, init_time=None,
                    time_shift=None, strict_overplot=False, terse=False, mon_str='', fig_files=None,
                    fig_list=None, show_killer_=True, hardcopy=False, compare_run_ver=True,
                    shift_soc_s=True):

    print(f"\n compare_run_sim: \
    \n{data_file=} \
    \n{unit_key=} \
    \n{time_end=} \
    \n{compare_run_ver=} \
    \n{shift_soc_s=} \
    \n{plots=} \
    \n{use_mon_soc_=} \
    \n{verbose=} \
    \n{scale_batt=} \
    \n{slr_hys_sim=} \
    \n{request_history=} \
    \n{init_time=} \
    \n{time_shift=} \
    \n{strict_overplot=} \
    \n{terse=} \
    \n{hardcopy=} \
    \n{mon_str=} \
    \n")

    if fig_files is None:
        fig_files = []
    if fig_list is None:
        fig_list = []

    mon_ver = None
    sim_ver = None
    sim_s_ver = None
    mon = None
    sim = None
    filename = None


    date_time = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    date_ = datetime.now().strftime("%y%m%d")

    # Transient  inputs
    zero_zero = False
    use_vb_sim = False
    use_ib_mon = False
    cc_dif_tol = 0.2
    legacy = False
    use_vb_raw = False
    dvoc_sim = 0.
    dvoc_mon = Dw
    use_mon_soc = use_mon_soc_

    # detect running interactively
    # this is written to run in pwd of call
    if data_file is None:
        path_to_data = easygui.fileopenbox(msg="choose your data file to plot")
        data_file = easygui.filesavebox(msg="pick new file name, cancel to keep", title="get new file name")
        if data_file is None:
            data_file = path_to_data
        else:
            os.rename(path_to_data, data_file)
        unit_key = easygui.enterbox(msg="enter pro0p, pro1a, soc0p, soc1a", title="get unit_key", default="pro1a")

    # Folder operations
    version = version_from_data_file(data_file)
    _, save_pdf_path, _ = local_paths(version)

    # # Load mon v4 (old)
    mon_run, sim_run, f, data_file_clean, temp_flt_file_clean, _ = \
        load_data(data_file, 1, unit_key, zero_zero, time_end, legacy=legacy, init_time=init_time,
                  time_shift=time_shift, mon_str=mon_str)
    sim_s_run = None

    # How to initialize
    if mon_run is not None:
        mon_run = rename_all(mon_run)

        # New run
        replicateOptions = UserOptions(mon_run=mon_run, sim_run=sim_run, run_type='RunSim', init_time=mon_run.init_time,
                                       use_ib_mon=use_ib_mon, use_mon_soc=use_mon_soc, use_vb_raw=use_vb_raw,
                                       add_voc_sim=dvoc_sim, add_voc_mon=dvoc_mon, use_vb_sim=use_vb_sim,
                                       verbose=verbose, scale_batt=scale_batt, slr_hys_sim=slr_hys_sim,
                                       request_history=request_history)
        mon_ver, sim_ver, sim_s_ver, mon, sim, Battery = replicate(replicateOptions)
        pass

        # Check if replicate broke early due to skip
        if mon_ver is None:
            print("\nCompareRunSim: Replication broke early due to data skip. Aborting without plots.")
            if show_killer_:
                tkinter.messagebox.showerror(title="Data Integrity Error",
                                             message="CompareRunSim: Replication broke early due to data skip.\n\nAborting without plots.")
            return fig_list, fig_files

        # C++ prints cc_dif from the previous cycle (Fault.cc_diff() is set in
        # sense_synth_select before monitor() updates soc_ekf and soc), so the
        # replicate's cc_dif runs one sample ahead.  Shift to align with the run.
        mon_ver = shift_time(mon_ver, 1, fields=('cc_dif',))

    # Save all time-dependent struct data to CSV files in the temp folder
    # if hardcopy and plots:
    if True:
        filename_root = data_file_clean.replace('.csv', '')
        if filename_root is None:
            print("save_struct_to_csv: no filename available, skipping CSV export")
        else:
            # Shift time in sim_ver; soc_s is computed at G.i but save_s() records at t[G.i-1]
            sim_ver = shift_time(sim_ver, 1)
            if shift_soc_s:
                sim_ver = shift_time(sim_ver, 1, fields=('soc_s',))
            for obj, struct_name in (
                (mon_run,   'mon_run'),
                (mon_ver,   'mon_ver'),
                (sim_run,   'sim_run'),
                (sim_ver,   'sim_ver'),
            ):
                save_struct_to_csv(obj, filename_root + '_' + struct_name + '.csv')

    # Plots
    if plots:
        if data_file_clean is not None:
            dir_root_test, data_root_test = str(PurePosixPath(data_file_clean).parent), PurePosixPath(data_file_clean).name
            data_root_test = data_root_test.replace('.csv', '')
            aug_file = PurePosixPath(data_file_clean).name.replace('.csv', '_') + PurePosixPath( Path(__file__).as_posix()).stem
        else:
            dir_root_test, data_root_test = str(PurePosixPath(temp_flt_file_clean).parent), PurePosixPath(temp_flt_file_clean).name
            data_root_test = data_root_test.replace('.csv', '')
            aug_file = PurePosixPath(temp_flt_file_clean).name.replace('.csv', '_') + PurePosixPath( Path(__file__).as_posix()).stem
        filename = str(PurePosixPath(save_pdf_path) / aug_file)
        plot_title = dir_root_test + '/' + data_root_test + '   ' + date_time

        S = PlotOptions(terse=terse, save_plots=hardcopy)
        if not S.terse and f is not None and temp_flt_file_clean and len(f.time_ux) > 1 and not strict_overplot:
            fig_list, fig_files = over_fault(f, filename, fig_files=fig_files, plot_title=plot_title, subtitle='faults',
                                             fig_list=fig_list, cc_dif_tol=cc_dif_tol, save_plots=S.save_plots)

        if mon_run is None and show_killer_:
            tkinter.messagebox.showwarning(message="CompareRunSim:  Data missing.  See monitor window for info.")
            # return None, None, None, None, None, None

        else:
            fig_list, fig_files = dom_plot(mon_run, mon_ver, sim_run, sim_ver, sim_s_run, sim_s_ver, filename, fig_files,
                                           plot_title=plot_title, fig_list=fig_list, strict_overplot=strict_overplot,
                                           terse=S.terse, run_type='RunSim', save_plots=S.save_plots)

        print('showing plots...')
        plt.ion()
        plt.show(block=False)

        # Copies — batch/AUTO mode only (no show_killer); show_killer's do_hardcopy handles the interactive case
        if S.save_plots and not show_killer_:
            import threading
            def _assemble(base=filename, path=save_pdf_path, dt=date_time):
                try:
                    precleanup_fig_files(output_pdf_name=base, path_to_pdfs=path)
                    print('\ncreating pdf...')
                    pngs_to_pdf(png_folder=path, output_pdf=base + '_' + dt + '.pdf')
                except Exception as e:
                    print(f"pdf assembly ERROR: {e}")
            threading.Thread(target=_assemble, daemon=True).start()

    # compare_run_ver runs regardless of plots; makes plots only when plots=True
    if compare_run_ver:
        try:
            from CompareRunVer import temp_folder, find_pairs, compare_pair, \
                report as ver_report, plot_diffs as ver_plot_diffs
            _ver_version = version_from_data_file(data_file) if data_file else None
            _case_stem = Path(data_file_clean).stem if data_file_clean else None
            if _ver_version and _case_stem:
                _ver_temp = temp_folder(_ver_version)
                if Path(_ver_temp).is_dir():
                    _ver_pairs = find_pairs(_ver_temp)
                    _ver_pairs = [(r, v) for r, v in _ver_pairs
                                  if ('_mon_run' in r.name or '_sim_run' in r.name)
                                  and r.name.startswith(_case_stem)]
                    if _ver_pairs:
                        _tol = 1e-3
                        _rtol = 1e-3  # ~3 significant digits — keeps q_capacity-class large-magnitude signals from tripping
                        _ver_results = [compare_pair(r, v, _tol, _rtol) for r, v in _ver_pairs]
                        ver_report(_ver_results, _tol, _rtol)
                        if plots:
                            _ver_ret = ver_plot_diffs(_ver_results, data_file=data_file, show_killer_=False)
                            if _ver_ret:
                                _ver_figs, _ver_files, _ = _ver_ret
                                fig_list.extend(_ver_figs)
                                fig_files.extend(_ver_files)
        except Exception as _ver_e:
            print(f"compare_run_ver step: {_ver_e}")

    if plots:
        string = 'plots ' + str(fig_list[0].number) + ' - ' + str(fig_list[-1].number)
        if show_killer_:
            show_killer(string, 'CompareRunSim', fig_list=fig_list, fig_files=fig_files, pdf_path=save_pdf_path,
                        pdf_base=filename, hardcopy=hardcopy)
        cleanup_fig_files(fig_files)
    print('DONE')

    return fig_list, fig_files


# noinspection PyUnusedLocal
def main():  # Example usage.  ok on 20260217
    if sys.platform == 'linux':
        gdrive = '/home/daveg/gdrive/'
    else:
        gdrive = 'G:/My Drive/'

    # Cut-pasted from GUI_TestSOC Run window
    """Request history:
        1:  ekf
        2:  soc
        3:  soc_s
        4:  temp
        5:  volt all
        6:  kf
        7:  dyn_m
        8:  vb_wrap
        9:  dyn_n
    """
    data_file = '/home/daveg/gdrive/GitHubArchive/SOC_Particle/dataReduction/g20250612a/vcFlat_soc3p2_hi_lo_bb.csv'
    unit_key = 'g20250612a_soc3p2_hi_lo_bb'
    time_end = None
    compare_run_ver = True
    shift_soc_s = True
    plots = True
    use_mon_soc_ = False
    verbose = False
    scale_batt = 1.0
    slr_hys_sim = 1.0
    request_history = 5
    init_time = None
    time_shift = None
    strict_overplot = True
    terse = True
    hardcopy = False
    mon_str = ''

    compare_run_sim(data_file=data_file, unit_key=unit_key, plots=plots, time_end=time_end,
                    use_mon_soc_=use_mon_soc_, verbose=verbose, scale_batt=scale_batt, slr_hys_sim=slr_hys_sim,
                    request_history=request_history, init_time=init_time, time_shift=time_shift,
                    strict_overplot=strict_overplot, terse=terse, hardcopy=hardcopy, compare_run_ver=compare_run_ver,
                    shift_soc_s=shift_soc_s)


# import cProfile
# if __name__ == '__main__':
#     cProfile.run('main()')
#


if __name__ == '__main__':  #
    main()
