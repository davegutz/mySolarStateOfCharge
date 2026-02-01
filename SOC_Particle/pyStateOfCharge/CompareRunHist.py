# CompareRunHist.py:  combine a CompareRunSim with CompareHistSim
# Copyright (C) 2024 Dave Gutz
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

from unite_pictures import unite_pictures_into_pdf, cleanup_fig_files, precleanup_fig_files
from DataOverModel import dom_plot
import matplotlib.pyplot as plt
from PlotKiller import show_killer
from CompareRunSim import compare_run_sim
from CompareHistSim import compare_hist_sim
from datetime import datetime
import os
import tkinter.messagebox
from local_paths import version_from_data_file, local_paths

import sys
if sys.platform == 'darwin':
    import matplotlib
    matplotlib.use('tkagg')
plt.rcParams['axes.grid'] = True
plt.rcParams['legend.fontsize'] = 'small'

# Suppress all UserWarning messages
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def compare_run_hist(data_file=None, unit_key=None, time_end=None, data_only=True, strict_overplot=False,
                     terse=False):

    print(f"\ncompare_run_hist:\n{data_file=}\n{data_only=}\n{unit_key=}\n{time_end=}\n{strict_overplot=}\n{terse=}\n")

    date_time = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    # date_ = datetime.now().strftime("%y%m%d")

    dfcs, mo_r, so_r, mv_r, sv_r, ssv_r =\
        compare_run_sim(data_file=data_file, unit_key=unit_key, time_end_in=time_end, data_only=data_only,
                        mon_str='', strict_overplot=strict_overplot, terse=terse)
    mo_h, so_h, mv_h, sv_h, ssv_h =\
        compare_hist_sim(data_file=data_file, unit_key=unit_key, time_end_in=time_end, data_only=data_only,
                         mon_t=True, sync_time=mo_r.time_run, strict_overplot=strict_overplot, terse=terse)

    # Plots
    if mo_r is not None and mo_h is not None:
        fig_list = []
        fig_files = []

        # File path operations
        version = version_from_data_file(data_file)
        _, save_pdf_path, _ = local_paths(version)

        (data_file_folder, _) = os.path.split(data_file)

        data_root_run = dfcs.split('/')[-1].replace('.csv', '')
        dir_root_run = data_file_folder.split('/')[-1].split('\\')[-1]
        filename = data_root_run + '__hist'

        # Plots
        plot_title = dir_root_run + '/' + data_root_run + '   ' + date_time

        fig_list, fig_files = dom_plot(mo_r, mo_h, so_r, so_h, ssv_h, filename, fig_files,
                                       plot_title=plot_title, fig_list=fig_list,
                                       run_str='_run', ver_str='_hist')  # all over all

        # Copies
        precleanup_fig_files(output_pdf_name=filename, path_to_pdfs=save_pdf_path)
        unite_pictures_into_pdf(outputPdfName=filename+'-'+date_time+'.pdf', save_pdf_path=save_pdf_path,
                                listWithImagesExtensions=["png"])
        cleanup_fig_files(fig_files)
        plt.show(block=False)
        string = 'plots ' + str(fig_list[0].number) + ' - ' + str(fig_list[-1].number)
        show_killer(string, 'CompareRunRun', fig_list=fig_list)

        return True
    else:
        tkinter.messagebox.showwarning(message="One or more sets of data missing.  See monitor window for info.")
        return False


def main():
    data_file = 'G:/My Drive/GitHubArchive/SOC_Particle/dataReduction\\g20250612a\\ampHiFail_soc2p2_hi_lo_bb.csv'
    data_only = True
    unit_key = 'g20250612a_soc2p2_hi_lo_bb'
    time_end = None
    terse = True
    strict_overplot = False

    compare_run_hist(data_file=data_file, unit_key=unit_key, time_end=time_end, data_only=data_only,
                     terse=terse, strict_overplot=strict_overplot)


if __name__ == '__main__':
    main()
