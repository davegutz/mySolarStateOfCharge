# MonSim:  Monitor and Simulator replication of Particle Photon Application
# Copyright (C) 2023 Dave Gutz
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
import sys

from MonSim import replicate, save_clean_file, UserOptions
from unite_pictures import unite_pictures_into_pdf, cleanup_fig_files, precleanup_fig_files
from CompareFault import over_fault
import matplotlib.pyplot as plt
from datetime import datetime
from load_data import load_data
from DataOverModel import dom_plot
import easygui
from PlotKiller import show_killer
import tkinter.messagebox
from local_paths import version_from_data_file, local_paths
import os
import plot.gp as gp

if sys.platform == 'darwin':
    import matplotlib
    matplotlib.use('tkagg')
plt.rcParams['axes.grid'] = True
plt.rcParams['legend.fontsize'] = 'small'

# Suppress all UserWarning messages
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def compare_run_sim(data_file=None, unit_key=None, time_end_in=None, plots=True, Dw=0.,  use_mon_soc_=False,
                    verbose=True, scale_in=1., slr_hys_sim=1., request_history=5, Battery=None, init_time_in=None,
                    time_shift_in=None, strict_overplot=False, terse=False, mon_str=''):

    if data_file.count('soc4p2_hi_lo'):
       IB_CHARGE_NOA = True
    else:
        IB_CHARGE_NOA = False

    print(f"\n \
compare_run_sim:\n{data_file=}\n{unit_key=}\n{time_end_in=}\n{plots=}\n{use_mon_soc_=}\n \
{IB_CHARGE_NOA=}\n{verbose=}\n{scale_in=}\n{slr_hys_sim=}\n{request_history=}\n{init_time_in=}\n{time_shift_in=}\n \
{strict_overplot=}\n{terse=}\n{mon_str=}\n \
          ")

    date_time = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    date_ = datetime.now().strftime("%y%m%d")

    # Transient  inputs
    zero_zero_in = False
    use_vb_sim_in = False
    use_ib_mon_in = False
    tune_in = False
    cc_dif_tol_in = 0.2
    legacy_in = False
    add_s_voc_soc_in = 0.
    use_vb_raw = False
    dvoc_sim_in = 0.
    dvoc_mon_in = Dw
    use_mon_soc_in = use_mon_soc_
    s_hys_sim_in = slr_hys_sim

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
        load_data(data_file, 1, unit_key, zero_zero_in, time_end_in, legacy=legacy_in, init_time_in=init_time_in,
                  time_shift_in=time_shift_in, mon_str=mon_str)

    # How to initialize
    if mon_run is None:
        tkinter.messagebox.showwarning(message="CompareRunSim:  Data missing.  See monitor window for info.")
        return None, None, None, None, None, None

    # New run
    mon_file_save = data_file_clean.replace(".csv", "_rep.csv")
    replicateOptions = UserOptions(mon_run=mon_run, sim_run=sim_run, run_type='RunSim', init_time=mon_run.init_time,
                                   use_ib_mon=use_ib_mon_in, use_mon_soc=use_mon_soc_in, use_vb_raw=use_vb_raw,
                                   add_voc_sim=dvoc_sim_in, add_voc_mon=dvoc_mon_in, use_vb_sim=use_vb_sim_in,
                                   add_s_voc_soc=add_s_voc_soc_in, verbose=verbose, scale_in=scale_in,
                                   slr_hys_sim=s_hys_sim_in, request_history=request_history,
                                   IB_CHARGE_NOA=IB_CHARGE_NOA)
    mon_ver, sim_ver, sim_s_ver, mon, sim, Battery = replicate(replicateOptions)
    pass
    save_clean_file(mon_ver, mon_file_save, 'mon_rep' + date_)

    # Plots
    if plots:
        fig_list = []
        fig_files = []
        dir_root_test, data_root_test = os.path.split(data_file_clean)
        data_root_test = data_root_test.replace('.csv', '')
        filename = data_root_test
        plot_title = dir_root_test + '/' + data_root_test + '   ' + date_time
        if not terse and f is not None and temp_flt_file_clean and len(f.time_ux) > 1 and not strict_overplot:
            fig_list, fig_files = over_fault(f, filename, fig_files=fig_files, plot_title=plot_title, subtitle='faults',
                                             fig_list=fig_list, cc_dif_tol=cc_dif_tol_in)
        if not terse:
            # fig_list, fig_files = ekf_plot(mon_run, mon_ver, sim_run, sim_ver, sim_s_ver, filename, fig_files,
            #                                plot_title=plot_title, fig_list=fig_list, run_str='',
            #                                ver_str='_ver', strict_overplot=strict_overplot)
            # fig_list, fig_files = sim_s_plot(mon_run, mon_ver, sim_run, sim_ver, sim_s_ver, filename, fig_files,
            #                                  plot_title=plot_title, fig_list=fig_list, run_str='',
            #                                  ver_str='_ver', strict_overplot=strict_overplot)
            # fig_list, fig_files = off_on_plot(mon_run, mon_ver, sim_run, sim_ver, sim_s_ver, filename, fig_files,
            #                                   plot_title=plot_title, fig_list=fig_list, run_str='',
            #                                   ver_str='_ver', strict_overplot=strict_overplot)
            if tune_in:
                fig_list, fig_files = gp.tune_r(mon_run, mon_ver, sim_s_ver, filename, fig_files,
                                             plot_title=plot_title, fig_list=fig_list, run_str='', ver_str='_ver')

        fig_list, fig_files = dom_plot(mon_run, mon_ver, sim_run, sim_ver, sim_s_ver, filename, fig_files,
                                       plot_title=plot_title, fig_list=fig_list, run_str='',
                                       ver_str='_ver', strict_overplot=strict_overplot, terse=terse,
                                       run_type='RunSim')

        # Copies
        precleanup_fig_files(output_pdf_name=filename, path_to_pdfs=save_pdf_path)
        unite_pictures_into_pdf(outputPdfName=filename+'_'+date_time+'.pdf', save_pdf_path=save_pdf_path)
        cleanup_fig_files(fig_files)
        plt.show(block=False)
        string = 'plots ' + str(fig_list[0].number) + ' - ' + str(fig_list[-1].number)
        show_killer(string, 'CompareRunSim', fig_list=fig_list)

    return data_file_clean, mon_run, sim_run, mon_ver, sim_ver, sim_s_ver


def main():

    import sys
    if sys.platform == 'linux':
        gdrive = '/home/daveg/gdrive/'
    else:
        gdrive = 'G:/My Drive/'

    data_file = 'G:/My Drive/GitHubArchive/SOC_Particle/dataReduction\\g20250612a\\offSitHysBmsBB_soc2p2_hi_lo_bb.csv'
    unit_key = 'g20250612a_soc2p2_hi_lo_chg'
    time_end_in = None
    plots = True
    use_mon_soc_ = False
    IB_CHARGE_NOA = False
    verbose = True
    scale_in = 1.0
    slr_hys_sim = 1.0
    request_history = 3
    init_time_in  = None  # that logic doesn't work yet
    time_shift_in = None
    strict_overplot_in = True

    # data_file = 'G:/My Drive/GitHubArchive/SOC_Particle/dataReduction\\g20250612a\\ssnoisenewamps_soc2p2_hi_lo_bb.csv' # problems with Vb=0 icharge=0
    # data_file = 'G:/My Drive/GitHubArchive/SOC_Particle/dataReduction\\g20250612a\\ssnoisenewampsXm2_soc2p2_hi_lo_bb.csv'  # problems with sat
    # data_file = 'G:/My Drive/GitHubArchive/SOC_Particle/dataReduction\\g20250612a\\ssnoisenewampsXm2Cap5_soc2p2_hi_lo_bb.csv'

    # Hardware
    # Xm0, Ca.5, DP1,
    # Rf, vv4,
    # Rk,
    # vv0,
    data_file = 'G:/My Drive/GitHubArchive/SOC_Particle/dataReduction\\g20250612a\\zero_with_pc_soc3p2_hi_lo_bb.csv'
    unit_key = 'g20250612a_soc3p2_hi_lo_bb'

    # # gdrive = '/home/daveg/Documents/'
    # # data_file = gdrive + 'vv4 20250905am_soc4p2_hi_lo_bb.csv'
    #
    # # unit_key = 'g20250612a_soc4p2_hi_lo_bb'  # old runsim work ******************
    # unit_key = 'g20250612a_soc2p2_hi_lo_chg'
    #
    # # The following are not implemented in GUI

    time_end_in = None
    # time_end_in = 6

    time_shift_in = None
    # time_shift_in = -1.811

    s_hys_sim_in = 1.
    # s_hys_sim_in = 0.

    verbose_in = False
    scale_in = 1.0

    # RunSim plot selection
    # 1=ekf   2=soc  3=soc_s  4=temp   5=volt  6=kf   7=dyn_m  8=vb_wrap
    request_hist_in = 3
    # request_hist_in = None

    # # mon_soc_in = False # old runsim work ******************
    use_mon_soc_ = False
    # use_mon_soc_ = True

    # plots = False
    plots = True

    terse_in = False
    # terse_in = True

    strict_overplot_in = False
    # strict_overplot_in = True

    compare_run_sim(data_file=data_file, unit_key=unit_key, plots=plots, time_end_in=time_end_in,
                    use_mon_soc_=use_mon_soc_, verbose=verbose_in, scale_in=scale_in, slr_hys_sim=s_hys_sim_in,
                    request_history=request_hist_in, init_time_in=init_time_in, time_shift_in=time_shift_in,
                    strict_overplot=strict_overplot_in, terse=terse_in)


# import cProfile
# if __name__ == '__main__':
#     cProfile.run('main()')
#


if __name__ == '__main__':
    main()
