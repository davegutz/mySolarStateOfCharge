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

from DataOverModel import dom_plot
from CompareFault import over_fault
from unite_pictures import cleanup_fig_files, precleanup_fig_files, pngs_to_pdf
import matplotlib.pyplot as plt
from datetime import datetime
from PlotKiller import show_killer
from pathlib import Path, PurePosixPath
from load_data import load_data, calculate_master_sync
from local_paths import version_from_data_path, local_paths
from plot.PlotOptions import  PlotOptions

# Suppress all UserWarning messages
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# noinspection PyPep8Naming
def compare_run_run(keys=None, data_file_folder_run=None, data_file_folder_test=None, sync_to_ctime=False,
                    terse=True, hardcopy=False):

    print(f"\ncompare_run_run:\n{keys=}\n{data_file_folder_run=}\n{data_file_folder_test=}\n{sync_to_ctime=}\n{terse=}\n{hardcopy=}\n")

    date_time = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    # date_ = datetime.now().strftime("%y%m%d")

    # Transient  inputs
    zero_zero = False
    time_end = None

    # Regression suite
    data_file_txt_run = keys[0][0]
    unit_key_run = keys[0][1]
    data_file_txt_test = keys[1][0]
    unit_key_test = keys[1][1]

    # Folder operations
    version = version_from_data_path(data_file_folder_test)
    _, save_pdf_path, _ = local_paths(version)

    # Load old ref data
    data_file_run = str(PurePosixPath(data_file_folder_run) / data_file_txt_run)
    mon_run, sim_run, f_run, data_file_run_clean, temp_flt_file_run_clean, sync_info_run = \
        load_data(data_file_run, 1, unit_key_run, zero_zero, time_end)
    sim_s_run = None
    mon_run.str_ = 'r1'
    sim_run.str_ = 's1'
    f_run.str_ = 'f1'

    # Load new test data
    data_file_test = str(PurePosixPath(data_file_folder_test) / data_file_txt_test)
    mon_test, sim_test, f_test, data_file_ver_clean, temp_flt_file_ver_clean, sync_info_test = \
        load_data(data_file_test, 1, unit_key_test, zero_zero, time_end)
    sim_s_test = None
    mon_test.str_ = 'r2'
    sim_test.str_ = 's2'
    f_test.str_ = 'f2'

    # Synchronize
    # Time since beginning of data to sync pulses
    if not sync_info_run.is_empty and not sync_info_test.is_empty and \
            sync_info_run.length == sync_info_test.length and (sync_info_run.length > 0 or sync_to_ctime is True):
        # Make target sync vector
        master_sync_del = calculate_master_sync(sync_info_run.del_mon, sync_info_test.del_mon)
        sync_info_run.synchronize(master_sync_del)
        mon_run.time = sync_info_run.time_mon.copy()
        sync_info_test.synchronize(master_sync_del)
        mon_test.time = sync_info_test.time_mon.copy()
        # print(f"{sync_to_ctime=}\n{sync_info_run.del_mon=}\n{sync_info_test.del_mon=}\n{master_sync_del=}\n{mon_test.time=}")
    elif sync_to_ctime:
        cTime_0_run = mon_run.cTime[0]
        cTime_sync = cTime_0_run
        mon_run.time = mon_run.cTime - cTime_sync
        mon_test.time = mon_test.cTime - cTime_sync
    else:
        print(f"Using simplified sync with |Ib|>0.  Data sets too small to sync or not equivalent number of sync pulses")

    # Plots
    fig_list = []
    fig_files = []
    dir_root_run, data_root_run = str(PurePosixPath(data_file_run_clean).parent), PurePosixPath(data_file_run_clean).name
    data_root_run = data_root_run.replace('.csv', '')
    dir_root_test, data_root_test = str(PurePosixPath(data_file_ver_clean).parent), PurePosixPath(data_file_ver_clean).name
    data_root_test = data_root_test.replace('.csv', '')

    filename = data_root_run + '__' + data_root_test + '_' + PurePosixPath( Path(__file__).as_posix()).stem
    filename = str(PurePosixPath(save_pdf_path) / filename)
    plot_title = dir_root_run + '/' + data_root_run + '__' + dir_root_test + '/' + data_root_test + '   ' + date_time

    S = PlotOptions(terse=terse, save_plots=hardcopy)

    if not S.terse:
        if temp_flt_file_run_clean and len(f_run.time_ux) > 1:
            fig_list, fig_files = over_fault(f_run, filename, fig_files=fig_files, plot_title=plot_title, subtitle='faults',
                                             fig_list=fig_list, save_plots=S.save_plots)

        if temp_flt_file_ver_clean and len(f_test.time_ux) > 1:
            fig_list, fig_files = over_fault(f_test, filename, fig_files=fig_files, plot_title=plot_title, subtitle='faults',
                                             fig_list=fig_list, save_plots=S.save_plots)

    fig_list, fig_files = dom_plot(mon_run, mon_test, sim_run, sim_test, sim_s_run, sim_s_test, filename, fig_files,
                                   plot_title=plot_title, fig_list=fig_list, run_type='RunRun', terse=S.terse,
                                   save_plots=S.save_plots)  # all over all

    print('showing plots...')
    plt.ion()
    plt.show(block=False)

    string = 'plots ' + str(fig_list[0].number) + ' - ' + str(fig_list[-1].number)
    show_killer(string, 'CompareRunRun', fig_list=fig_list, fig_files=fig_files, pdf_path=save_pdf_path, pdf_base=filename, hardcopy=hardcopy)
    cleanup_fig_files(fig_files)
    print('DONE')

    return fig_list, fig_files


# noinspection PyUnusedLocal
def main():
    import sys
    if sys.platform == 'linux':
        gdrive = '/home/daveg/gdrive/'
    else:
        gdrive = 'G:/My Drive/'

    # Cut-pasted from GUI_TestSOC Run window
    keys = [('ampHiEmptFail_soc2p2_hi_lo_bb.csv', 'g20250612a_soc2p2_hi_lo_bb'),
            ('ampHiEmptFail_soc3p2_hi_lo_bb.csv', 'g20250612a_soc3p2_hi_lo_bb')]
    data_file_folder_run = 'G:/My Drive/GitHubArchive/SOC_Particle/dataReduction/g20250612a'
    data_file_folder_test = 'G:/My Drive/GitHubArchive/SOC_Particle/dataReduction/g20250612a'
    sync_to_ctime = False
    terse = True
    hardcopy = False

    compare_run_run(keys=keys, data_file_folder_run=data_file_folder_run, data_file_folder_test=data_file_folder_test,
                    sync_to_ctime=sync_to_ctime, terse=terse, hardcopy=hardcopy)


if __name__ == '__main__':  # Example usage.  Ran ok 20260217
    main()
