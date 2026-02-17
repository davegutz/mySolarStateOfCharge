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
from unite_pictures import unite_pictures_into_pdf, cleanup_fig_files, precleanup_fig_files
from datetime import datetime
from local_paths import version_from_data_file, local_paths
import os
from CompareHistSim import load_hist_and_prep
from CompareFault import overall_fault, over_fault

import sys
if sys.platform == 'darwin':
    import matplotlib
    matplotlib.use('tkagg')
plt.rcParams['axes.grid'] = True
plt.rcParams['legend.fontsize'] = 'small'

# Suppress all UserWarning messages
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def compare_hist_hist(data_file_run=None, unit_key_run=None, data_file_tst=None, unit_key_tst=None,
                      dt_resample=10, plots=True):

    print(f"\ncompare_hist_sim:\n{data_file_run=}\n{unit_key_run=}\n{data_file_tst=}\n{unit_key_tst=}\n{dt_resample=}\n")

    date_time = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # Save these
    cc_dif_tol_in = 0.2
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
    _, data_file_txt = os.path.split(data_file_run)
    version = version_from_data_file(data_file_run)
    path_to_temp, save_pdf_path, _ = local_paths(version)

    # Plots
    if plots:
        fig_list = []
        fig_files = []
        plot_title = filename_run + filename_tst + '   ' + date_time
        if fault_run is not None and len(fault_run.time) > 1:
            fig_list, fig_files = over_fault(fault_run, filename_run, fig_files=fig_files, plot_title=plot_title,
                                             subtitle='faults_run', fig_list=fig_list, cc_dif_tol=cc_dif_tol_in,
                                             time_units='sec')
        if fault_tst is not None and len(fault_tst.time) > 1:
            fig_list, fig_files = over_fault(fault_tst, filename_tst, fig_files=fig_files, plot_title=plot_title,
                                             subtitle='faults_tst', fig_list=fig_list, cc_dif_tol=cc_dif_tol_in,
                                             time_units='sec')
        if hist_20C_run is not None and len(hist_20C_run.time) > 1:
            sim_run = None
            plot_init_in = False
            fig_list, fig_files = dom_plot(mon_run, mon_tst, sim_run, sim_tst, sim_s_tst, filename_run, fig_files,
                                           plot_title=plot_title, fig_list=fig_list, run_str='_'+unit_run,
                                           ver_str='_'+unit_tst, run_type='HistHist')
            fig_list, fig_files = overall_fault(mon_run, mon_tst, sim_tst, sim_s_tst, filename_run,
                                                fig_files, plot_title=plot_title, fig_list=fig_list)

        precleanup_fig_files(output_pdf_name=filename_run, path_to_pdfs=save_pdf_path)
        unite_pictures_into_pdf(outputPdfName=filename_run+'_'+date_time+'.pdf', save_pdf_path=save_pdf_path)
        cleanup_fig_files(fig_files)
    
        plt.show(block=False)
        string = 'plots ' + str(fig_list[0].number) + ' - ' + str(fig_list[-1].number)
        show_killer(string, 'CompareFault', fig_list=fig_list)

    return mon_run, sim_run, mon_tst, sim_tst, sim_s_tst


def main():

    # User inputs (multiple input_files allowed
    data_file_run = 'G:/My Drive/GitHubArchive/SOC_Particle/dataReduction\\g20250612a\\rapidTweakRegression_soc2p2_hi_lo_bb.csv'
    unit_key_run = 'g20250612a_soc2p2_bb'
    data_file_tst = 'G:/My Drive/GitHubArchive/SOC_Particle/dataReduction\\g20250612a\\rapidTweakRegression_soc3p2_hi_lo_bb.csv'
    unit_key_tst = 'g20250612a_soc3p2_bb'
    dt_resample = 10

    # Do this when running compare_hist_sim on run that schedule extracted assuming constant Tb
    # Tb_force = 35

    compare_hist_hist(data_file_run=data_file_run, unit_key_run=unit_key_run,
                      data_file_tst=data_file_tst, unit_key_tst=unit_key_tst,
                      dt_resample=dt_resample)


if __name__ == '__main__':  # Example usage.  Ran ok 20260217
    main()
