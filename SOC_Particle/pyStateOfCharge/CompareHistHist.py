# CompareHistSim.py:  load fault, hist, summ data and compare to simulation.
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

""" Slice and dice the history dumps."""

import matplotlib.pyplot as plt
from PlotKiller import show_killer
from DataOverModel import dom_plot
from unite_pictures import cleanup_fig_files, precleanup_fig_files, pngs_to_pdf
from datetime import datetime
from local_paths import version_from_data_file, local_paths
from pathlib import Path, PurePosixPath
from CompareHistSim import load_hist_and_prep
from CompareFault import overall_fault, over_fault
from plot.PlotOptions import PlotOptions

plt.rcParams['axes.grid'] = True
plt.rcParams['legend.fontsize'] = 'small'
plt.rcParams['savefig.dpi'] = 300  # Set default savefig DPI to 300
plt.rcParams['figure.dpi'] = 100  # Also increase display DPI for consistency

# Suppress all UserWarning messages
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# noinspection PyPep8Naming
def compare_hist_hist(data_file_run=None, unit_key_run=None, data_file_tst=None, unit_key_tst=None,
                      dt_resample=10, plots=True, terse=False, hardcopy=False):

    print(f"\ncompare_hist_hist:\n{data_file_run=}\n{unit_key_run=}\n{data_file_tst=}\n{unit_key_tst=}\n{dt_resample=}\n{terse=}\n{hardcopy=}\n")

    date_time = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # Save these
    cc_dif_tol = 0.2
    sim_s_run = None
    sim_s_tst = None

    # Load history, normalizing all soc and Tb to 20C
    mon_run, sim_run, unit_run, fault_run, hist_20C_run, filename_run, Battery_run = \
        load_hist_and_prep(data_file=data_file_run, unit_key=unit_key_run, dt_resample=dt_resample)
    mon_run.str = 'h1'
    mon_tst, sim_tst, unit_tst, fault_tst, hist_20C_tst, filename_tst, Battery_tst = \
        load_hist_and_prep(data_file=data_file_tst, unit_key=unit_key_tst, dt_resample=dt_resample)
    mon_tst.str = 'h2'

    # Synchronize
    d_time = mon_tst.time_ux[0] - mon_run.time_ux[0]
    if d_time > 0:
        mon_tst.time += d_time
    else:
        mon_run.time -= d_time

    # File path operations
    version = version_from_data_file(data_file_run)
    path_to_temp, save_pdf_path, _ = local_paths(version)

    # Plots
    if plots:
        filename_run  = filename_run.replace('CompareHistSim', '')
        filename_tst = filename_tst.replace('CompareHistSim', '')
        aug_file = filename_run + '__' + filename_tst + '_' + PurePosixPath( Path(__file__).as_posix()).stem
        filename = str(PurePosixPath(save_pdf_path) / aug_file)
        fig_list = []
        fig_files = []
        plot_title = filename_run + filename_tst + '   ' + date_time

        S = PlotOptions(terse=terse, save_plots=hardcopy)

        if fault_run is not None and len(fault_run.time) > 1:
            fig_list, fig_files = over_fault(fault_run, filename, fig_files=fig_files, plot_title=plot_title,
                                             subtitle='faults_run', fig_list=fig_list, cc_dif_tol=cc_dif_tol,
                                             time_units='sec', save_plots=S.save_plots)

        if fault_tst is not None and len(fault_tst.time) > 1:
            fig_list, fig_files = over_fault(fault_tst, filename, fig_files=fig_files, plot_title=plot_title,
                                             subtitle='faults_tst', fig_list=fig_list, cc_dif_tol=cc_dif_tol,
                                             time_units='sec', save_plots=S.save_plots)

        if hist_20C_run is not None and len(hist_20C_run.time) > 1:
            sim_run = None
            fig_list, fig_files = dom_plot(mon_run, mon_tst, sim_run, sim_tst, sim_s_run, sim_s_tst, filename_run, fig_files,
                                           plot_title=plot_title, fig_list=fig_list, run_type='HistHist', terse=S.terse,
                                           save_plots=S.save_plots)
            fig_list, fig_files = overall_fault(mon_run, mon_tst, sim_run, sim_tst, sim_s_run, sim_s_tst, filename_run,
                                                fig_files, plot_title=plot_title, fig_list=fig_list,
                                                run_type='HistHist', save_plots=S.save_plots)

        print('showing plots...')
        plt.ion()
        plt.show(block=False)
        string = 'plots ' + str(fig_list[0].number) + ' - ' + str(fig_list[-1].number)
        show_killer(string, 'CompareFault', fig_list=fig_list, fig_files=fig_files, pdf_path=save_pdf_path, pdf_base=filename, hardcopy=hardcopy)
        cleanup_fig_files(fig_files)
    print('DONE')

    return mon_run, sim_run, mon_tst, sim_tst, sim_s_tst


def main():

    # User inputs (multiple input_files allowed

    # Cut-pasted from GUI_TestSOC Run window
    data_file_run = 'G:/My Drive/GitHubArchive/SOC_Particle/dataReduction/g20250612a/ampHiEmptFail_soc2p2_hi_lo_bb.csv'
    unit_key_run = 'g20250612a_soc2p2_hi_lo_bb'
    data_file_tst = 'G:/My Drive/GitHubArchive/SOC_Particle/dataReduction/g20250612a/ampHiEmptFail_soc3p2_hi_lo_bb.csv'
    unit_key_tst = 'g20250612a_soc3p2_hi_lo_bb'
    dt_resample = 1
    terse = True
    hardcopy = False

    compare_hist_hist(data_file_run=data_file_run, unit_key_run=unit_key_run,
                      data_file_tst=data_file_tst, unit_key_tst=unit_key_tst,
                      dt_resample=dt_resample, terse=terse, hardcopy=hardcopy)


if __name__ == '__main__':  # Example usage.  Ran ok 20260217
    main()
