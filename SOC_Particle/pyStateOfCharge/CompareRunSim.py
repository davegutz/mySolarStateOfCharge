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


# noinspection PyPep8Naming
def compare_run_sim(data_file=None, unit_key=None, time_end=None, plots=True, Dw=0.,  use_mon_soc_=False,
                    verbose=True, scale_batt=1., slr_hys_sim=1., request_history=5, init_time=None,
                    time_shift=None, strict_overplot=False, terse=False, mon_str='', fig_files=None,
                    fig_list=None, show_killer_=True, hardcopy=False):

    print(f"\n compare_run_sim: \
    \n{data_file=} \
    \n{unit_key=} \
    \n{time_end=} \
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
            tkinter.messagebox.showerror(title="Data Integrity Error",
                                         message="CompareRunSim: Replication broke early due to data skip.\n\nAborting without plots.")
            return fig_list, fig_files

    # Save all time-dependent struct data to CSV files in the temp folder
    if hardcopy and plots:
        filename_root = data_file_clean.replace('.csv', '')
        if filename_root is None:
            print("save_struct_to_csv: no filename available, skipping CSV export")
        else:
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

        if mon_run is None:
            tkinter.messagebox.showwarning(message="CompareRunSim:  Data missing.  See monitor window for info.")
            # return None, None, None, None, None, None

        else:
            fig_list, fig_files = dom_plot(mon_run, mon_ver, sim_run, sim_ver, sim_s_run, sim_s_ver, filename, fig_files,
                                           plot_title=plot_title, fig_list=fig_list, strict_overplot=strict_overplot,
                                           terse=S.terse, run_type='RunSim', save_plots=S.save_plots)

        # Copies
        if S.save_plots and not S.terse:
            precleanup_fig_files(output_pdf_name=filename, path_to_pdfs=save_pdf_path)
            print('\ncreating pdf...')
            pngs_to_pdf(png_folder=save_pdf_path, output_pdf=filename + '_' + date_time + '.pdf')

        print('showing plots...')
        plt.ion()
        plt.show(block=False)

        string = 'plots ' + str(fig_list[0].number) + ' - ' + str(fig_list[-1].number)
        if show_killer_:
            show_killer(string, 'CompareRunSim', fig_list=fig_list, fig_files=fig_files, pdf_path=save_pdf_path, pdf_base=filename, hardcopy=hardcopy)
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
    data_file = '/home/daveg/gdrive/GitHubArchive/SOC_Particle/dataReduction/g20250612a/ampHiFail_soc3p2_hi_lo_bb.csv'
    unit_key = 'g20250612a_soc3p2_hi_lo_bb'
    time_end = None
    plots = True
    use_mon_soc_ = False
    verbose = True
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
                    strict_overplot=strict_overplot, terse=terse, hardcopy=hardcopy)


# import cProfile
# if __name__ == '__main__':
#     cProfile.run('main()')
#


if __name__ == '__main__':  #
    main()
