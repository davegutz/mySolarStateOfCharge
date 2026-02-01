# GP_batteryEKF - general purpose battery class for EKF use
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

"""Define a general purpose battery model including Randles' model and SoC-VOV model as well as Kalman filtering
support for simplified Mathworks' tracker. See Huria, Ceraolo, Gazzarri, & Jackey, 2013 Simplified Extended Kalman
Filter Observer for SOC Estimation of Commercial Power-Oriented LFP Lithium Battery Cells.
Dependencies:
    - numpy      (everything)
    - matplotlib (plots)
    - reportlab  (figures, pdf)
"""

from unite_pictures import unite_pictures_into_pdf, cleanup_fig_files
from local_paths import version_from_data_file, local_paths
from plot.PlotOptions import PlotOptions
from Battery import overall_batt
import matplotlib.pyplot as plt
from datetime import datetime
import plot.off_on as off_on
import plot.sim_s as sim_s
from Colors import Colors
import plot.dom as dom
import plot.ult as ult
import plot.gp as gp
import numpy as np
import sys
import re
import os

if sys.platform == 'darwin':
    import matplotlib
    matplotlib.use('tkagg')
plt.rcParams.update({'figure.max_open_warning': 0})


def dom_plot(mr, mv, sr, sv, smv, filename, fig_files=None, plot_title=None, fig_list=None, plot_init_in=False,
             run_str='_run', ver_str='_ver', strict_overplot=False, terse=False):
    print('dom_plot', end=':  ')
    if fig_files is None:
        fig_files = []

    if not terse:
        # fig_list, fig_files = hist.hs_plots(mr, mv, sr, sv, smv, filename, fig_files=fig_files, plot_title=plot_title, fig_list=fig_list,
        #                                     strict_overplot=strict_overplot)
        # fig_list, fig_files = hist.hs_tune_plots(mr, mv, sr, sv, smv, filename, fig_files=fig_files, plot_title=plot_title, fig_list=fig_list,
        #                                          strict_overplot=strict_overplot)
        fig_list, fig_files = dom.ekf_plots(mr, mv, sr, sv, smv, filename, fig_files=fig_files, plot_title=plot_title, fig_list=fig_list,
                                            strict_overplot=strict_overplot)
        if  plot_init_in and hasattr(smv, 'time') and hasattr(sr, 'time'):
            fig_list, fig_files = dom.init_1(mr, mv, sr, sv, smv, filename, fig_files=fig_files, plot_title=plot_title,
                                             fig_list=fig_list, strict_overplot=strict_overplot)
            fig_list, fig_files = dom.init_1a(mr, mv, sr, sv, smv, filename, fig_files=fig_files, plot_title=plot_title,
                                              fig_list=fig_list, strict_overplot=strict_overplot)
        fig_list, fig_files = sim_s.sim_s_plots(mr, mv, sr, sv, smv, filename, fig_files=fig_files, plot_title=plot_title, fig_list=fig_list,
                                                strict_overplot=strict_overplot)
        fig_list, fig_files = dom.dom_2(mr, mv, sr, sv, smv, filename, fig_files=fig_files, plot_title=plot_title, fig_list=fig_list,
                                    strict_overplot=strict_overplot)
        fig_list, fig_files = dom.dom_3(mr, mv, sr, sv, smv, filename, fig_files=fig_files, plot_title=plot_title, fig_list=fig_list,
                                    strict_overplot=strict_overplot)
        fig_list, fig_files = dom.dom_4(mr, mv, sr, sv, smv, filename, fig_files=fig_files, plot_title=plot_title, fig_list=fig_list,
                                    strict_overplot=strict_overplot)
        fig_list, fig_files = dom.dom_4a(mr, mv, sr, sv, smv, filename, fig_files=fig_files, plot_title=plot_title,
                                     fig_list=fig_list, strict_overplot=strict_overplot)

        figOptions = PlotOptions(mr=mr, mv=mv, sr=sr, sv=sv, smv=smv, filename=filename, plot_title=plot_title,
                                 strict_overplot=strict_overplot)

        fig_list, fig_files = gp.gp_1(figOptions, fig_files=fig_files, fig_list=fig_list)


        fig_list, fig_files = gp.gp_2(mr, mv, sr, sv, smv, filename, fig_files=fig_files, plot_title=plot_title,
                                      fig_list=fig_list, strict_overplot=strict_overplot)
        fig_list, fig_files = gp.gp_2_nn_lag(mr, mv, sr, sv, smv, filename, fig_files=fig_files, plot_title=plot_title,
                                      fig_list=fig_list, strict_overplot=strict_overplot)
        fig_list, fig_files = gp.gp_3_ekf(mr, mv, sr, sv, smv, filename, fig_files=fig_files, plot_title=plot_title,
                                          fig_list=fig_list, strict_overplot=strict_overplot)
        fig_list, fig_files = off_on.off_on_plots(mr, mv, sr, sv, smv, filename, fig_files, plot_title=plot_title,
                                                  fig_list=fig_list, strict_overplot=strict_overplot)
    fig_list, fig_files = gp.gp_3_tune(mr, mv, sr, sv, smv, filename, fig_files=fig_files, plot_title=plot_title,
                                       fig_list=fig_list, strict_overplot=strict_overplot)
    fig_list, fig_files = ult.ult_1(mr, mv, sr, sv, smv, filename, fig_files=fig_files, plot_title=plot_title,
                                fig_list=fig_list, strict_overplot=strict_overplot)
    return fig_list, fig_files


def count_text_fields(line):
    count = 0
    tokens = line.split(',')
    for token in tokens:
        try:
            float(token)
        except ValueError:
            # print(f"bad {token}")
            count += 1
    return count


def filter_f15_sequence(data_stream):
    # The escape sequence is typically ^[[28~, where ^[ is the ESC character (ASCII 27)
    # In a Python string, you can represent ESC as '\x1b' or '\033'
    f15_sequence = re.escape('\x1b[28~')
    # Use re.sub to replace the sequence with an empty string
    filtered_data = re.sub(f15_sequence, '', data_stream)
    return filtered_data


def write_clean_file(path_to_data, type_=None, hdr_key=None, unit_key=None, skip=1, comment_str='#'):
    """First line with hdr_key defines the number of fields to be imported cleanly"""
    import os
    (path, basename) = os.path.split(path_to_data)
    version = version_from_data_file(path_to_data)
    (path_to_temp, save_pdf_path, _) = local_paths(version)
    csv_file = path_to_temp+'/'+basename.replace('.csv', type_ + '.csv', 1)
    # Header
    have_header_str = None
    num_fields = 0
    with open(path_to_data, "r", encoding='cp437') as input_file:
        with open(csv_file, "w") as output:
            try:
                for line in input_file:
                    line = filter_f15_sequence(line)  # ESC[28~ injected by f15 keypress GUI_TestSOC to keep term awake
                    if line.__contains__('FRAG'):
                        print(Colors.fg.red, "\n\n\nDataOverModel(write_clean_file): Heap fragmentation error\
                         detected in Particle.  Decrease NSUM constant and re-run\n\n", Colors.reset)
                        return None
                    if line.__contains__(hdr_key):
                        if have_header_str is None:
                            have_header_str = True  # write one title only
                            output.write('skip,' + line)
                            num_fields = line.count(',')  # first line with hdr_key defines number of fields
            except IOError:
                print("DataOverModel381:", line)  # last line
    # Data
    num_lines = 0
    num_text_run = 0
    num_lines_in = 0
    num_skips = 0
    unit_key_found = False
    skipped_last = False
    with (open(path_to_data, "r", encoding='cp437') as input_file):  # reads all characters even bad ones
        with open(csv_file, "a") as output:
            for line in input_file:
                line = filter_f15_sequence(line)  # ESC[28~ injected by f15 keypress GUI_TestSOC to keep term awake
                if line.__contains__(unit_key) and not line.__contains__('Config:'):
                    unit_key_found = True
                    # if line.__contains__('946s868214.902'):
                    #     print("line_run:", line_run)
                    #     print("bad line:", line)
                    #     exit(1)
                    num_text = count_text_fields(line)
                    if num_lines == 0:
                        num_text_run = num_text
                    if line.count(",") == num_fields and line.count(";") == 0 and \
                            num_text == num_text_run and \
                            re.search(r'[^a-zA-Z0-9+-_.:, ]', line[:-1]) is None and \
                            (num_lines == 0 or ((num_lines_in+1) % skip) == 0) and line.count(comment_str) == 0:
                        output.write("{:2d},".format(skipped_last) + line)
                        num_lines += 1
                        skipped_last = False
                    else:
                        print(f"discarding: ", line, end='')
                        print(f"  line.count(',') == num_fields  {line.count(",") == num_fields}   \
\nAND num_text == num_text_run {num_text == num_text_run} \
\nAND re.search(r'[^a-zA-Z0-9+-_.:, ]', line[:-1]) is None {re.search(r'[^a-zA-Z0-9+-_.:, ]', line[:-1]) is None} \
\nAND (num_lines == 0 or ((num_lines_in+1) % skip) == 0) {(num_lines == 0 or ((num_lines_in+1) % skip) == 0)} \
\nAND line.count(comment_str) == 0 {line.count(comment_str) == 0}")
                        print(f"{line.count(',')=} {num_fields=}")
                        print(f"{line[-1]=}")
                        print(f"{num_text=} {num_text_run=}")
                        num_skips += 1
                        skipped_last = True
                    num_lines_in += 1
    if not num_lines:
        csv_file = None
        print("I(write_clean_file): no data to write")
        if not unit_key_found:
            print("W(write_clean_file):  unit_key not found in ", basename, ".  Looking with '{:s}'".format(unit_key))
    else:
        print("Wrote(write_clean_file):", csv_file, num_lines, "lines", num_skips, "skips", num_fields, "fields")
    return csv_file


if __name__ == '__main__':
    import sys
    import doctest

    doctest.testmod(sys.modules['__main__'])
    if sys.platform == 'darwin':
        import matplotlib
        matplotlib.use('tkagg')
    plt.rcParams['axes.grid'] = True
    plt.rcParams['legend.fontsize'] = 'small'


    def main(data_file_old_txt, unit_key):
        from MonSim import replicate, save_clean_file, save_clean_file_sim
        # Trade study inputs
        # i-->0 provides continuous anchor to reset filter (why?)  i shifts important --> 2 current sensors,
        #   hyst in ekf
        # saturation provides periodic anchor to reset filter
        # reset soc periodically anchor user display
        # tau_sd creating an anchor.   So large it's just a pass through
        # TODO:  temp sensitivities and mitigation

        # Config inputs
        # from MonSimNomConfig import *

        # Transient  inputs
        time_end = None
        zero_zero_in = False
        # time_end = 1500.

        # Load data (must end in .txt) txt_file, type, hdr_key, unit_key
        data_file_clean = write_clean_file(data_file_old_txt, type_='_mon', hdr_key='unit,',
                                           unit_key=unit_key)
        data_file_sim_clean = write_clean_file(data_file_old_txt, type_='_sim', hdr_key='unit_m',
                                               unit_key='unit_sim,')

        # Load
        mon_run_raw = np.genfromtxt(data_file_clean, delimiter=',', names=True, dtype=float).view(np.recarray)
        mon_run = SavedData(mon_run_raw, time_end, zero_zero=zero_zero_in, str_='')
        try:
            sim_run_raw = np.genfromtxt(data_file_sim_clean, delimiter=',', names=True, dtype=float).view(np.recarray)
            sim_run = SavedDataSim(mon_run.time_run, sim_run_raw, time_end, str_='_s')
        except IOError:
            sim_run = None

        # Run model
        from MonSim import UserOptions
        replicateOptions = UserOptions(mon_run=mon_run, init_time=1.)
        mon_ver, sim_ver, sim_s_ver = replicate(replicateOptions)
        date_ = datetime.now().strftime("%y%m%d")
        mon_file_save = data_file_clean.replace(".csv", "_rep.csv")
        save_clean_file(mon_ver, mon_file_save, '_mon_rep' + date_)
        if data_file_sim_clean:
            sim_file_save = data_file_sim_clean.replace(".csv", "_rep.csv")
            save_clean_file_sim(sim_s_ver, sim_file_save, '_sim_rep' + date_)

        # Plots
        fig_list = []
        fig_files = []
        date_time = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        filename = data_file_clean.split('/')[-1].replace('.csv', '-') + os.path.split(__file__)[1].split('.')[0]
        plot_title = filename + '   ' + date_time
        fig_list, fig_files = overall_batt(mon_ver, sim_ver, filename, fig_files, plot_title=plot_title,
                                           fig_list=fig_list, suffix='_ver')  # Could be confusing because sim over mon
        fig_list, fig_files = dom_plot(mon_run, mon_ver, sim_run, sim_ver, sim_s_ver, filename, fig_files,
                                       plot_title=plot_title, fig_list=fig_list, run_str='', ver_str='_ver')
        # fig_list, fig_files = tune_r(mon_run, mon_ver, sim_s_ver, filename, fig_files,
        #                           plot_title=plot_title, fig_list=fig_list, run_str='', ver_str='_ver')
        unite_pictures_into_pdf(outputPdfName=filename+'_'+date_time+'.pdf',
                                save_pdf_path='../dataReduction/figures')
        cleanup_fig_files(fig_files)

        plt.show()


    # python DataOverModel.py("../dataReduction/rapidTweakRegressionTest20220711.txt", "pro_2022")

    """
    PyCharm Sample Run Configuration Parameters (right click in pyCharm - Modify Run Configuration:
        "../dataReduction/slowTweakRegressionTest20220711.txt" "pro_2022"
        "../dataReduction/serial_20220624_095543.txt"    "pro_2022"
        "../dataReduction/real world rapid 20220713.txt" "soc0_2022"
        "../dataReduction/real world Xp20 20220715.txt" "soc0_2022"
    
    PyCharm Terminal:
    python DataOverModel.py "../dataReduction/serial_20220624_095543.txt" "pro_2022"
    python DataOverModel.py "../dataReduction/ampHiFail20220731.txt" "pro_2022"
    

    android:
    python Python/DataOverModel.py "USBTerminal/serial_20220624_095543.txt" "pro_2022"
    """

    if __name__ == "__main__":
        import sys
        print(sys.argv[1:])
        main(sys.argv[1], sys.argv[2])
