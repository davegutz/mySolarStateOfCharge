# GP_batteryEKF - general purpose battery class for EKF use
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

"""Define a general purpose battery model including Randles' model and SoC-VOV model as well as Kalman filtering
support for simplified Mathworks' tracker. See Huria, Ceraolo, Gazzarri, & Jackey, 2013 Simplified Extended Kalman
Filter Observer for SOC Estimation of Commercial Power-Oriented LFP Lithium Battery Cells.
Dependencies:
    - numpy      (everything)
    - matplotlib (plots)
    - reportlab  (figures, pdf)
"""

from local_paths import version_from_data_file, local_paths
from plot.PlotOptions import PlotOptions
import matplotlib.pyplot as plt
import plot.off_on as off_on
import plot.sim_s as sim_s
from Colors import Colors
import plot.dom as dom
import plot.ult as ult
import plot.gp as gp
import sys
import re
from pathlib import PurePosixPath

if sys.platform == 'darwin':
    import matplotlib
    matplotlib.use('tkagg')
plt.rcParams.update({'figure.max_open_warning': 0})


# noinspection PyPep8Naming
def dom_plot(mr, mv, sr, sv, smr, smv, filename, fig_files=None, plot_title=None, fig_list=None, plot_init=False,
             strict_overplot=False, terse=False, run_type=None, save_plots=False):

    print('dom_plot', end=':  ')
    if fig_files is None:
        fig_files = []
    figOptions = PlotOptions(mr=mr, mv=mv, sr=sr, sv=sv, smr=smr, smv=smv, filename=filename, plot_title=plot_title,
                             strict_overplot=strict_overplot, run_type=run_type, terse=terse, save_plots=save_plots)

    if not figOptions.terse:
        fig_list, fig_files = dom.ekf_plots(figOptions, fig_files=fig_files, fig_list=fig_list)
        if  plot_init and hasattr(smv, 'time') and hasattr(sr, 'time'):
            fig_list, fig_files = dom.init_1(figOptions, fig_files=fig_files, fig_list=fig_list)
            fig_list, fig_files = dom.init_1a(figOptions, fig_files=fig_files, fig_list=fig_list)
        fig_list, fig_files = sim_s.sim_s_plots(figOptions, fig_files=fig_files, fig_list=fig_list)
        fig_list, fig_files = dom.dom_2(figOptions, fig_files=fig_files, fig_list=fig_list)
        fig_list, fig_files = dom.dom_3(figOptions, fig_files=fig_files, fig_list=fig_list)
        fig_list, fig_files = dom.dom_4(figOptions, fig_files=fig_files, fig_list=fig_list)
        fig_list, fig_files = dom.dom_4a(figOptions, fig_files=fig_files, fig_list=fig_list)
        fig_list, fig_files = gp.gp_1(figOptions, fig_files=fig_files, fig_list=fig_list)
        fig_list, fig_files = gp.gp_2(figOptions, fig_files=fig_files, fig_list=fig_list)
        fig_list, fig_files = gp.gp_2_nn_lag(figOptions, fig_files=fig_files, fig_list=fig_list)
        fig_list, fig_files = gp.gp_3_ekf(figOptions, fig_files=fig_files, fig_list=fig_list)
        fig_list, fig_files = off_on.off_on_plots(figOptions, fig_files=fig_files, fig_list=fig_list)

    fig_list, fig_files = gp.gp_3_tune(figOptions, fig_files=fig_files, fig_list=fig_list)
    fig_list, fig_files = ult.ult_1(figOptions, fig_files=fig_files, fig_list=fig_list)

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
    path, basename = str(PurePosixPath(path_to_data).parent), PurePosixPath(path_to_data).name
    version = version_from_data_file(path_to_data)
    (path_to_temp, save_pdf_path, _) = local_paths(version)
    csv_file = path_to_temp+'/'+basename.replace('.csv', type_ + '_' + unit_key + '.csv', 1)
    # Header
    have_header_str = None
    num_fields = 0
    with open(path_to_data, "r", encoding='cp437') as input_file:  # reads all characters even bad ones
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
                            output.write('skip' + type_ + ',' + line)
                            num_fields = line.count(',')  # first line with hdr_key defines number of fields
            except IOError:
                print("DataOverModel381:", line)  # last line
            # Data - rewind to start of input, continue appending to same output
            input_file.seek(0)
            num_lines = 0
            num_text_run = 0
            num_lines = 0
            num_skips = 0
            unit_key_found = False
            skipped_last = False
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
                            (num_lines == 0 or ((num_lines+1) % skip) == 0) and line.count(comment_str) == 0:
                        output.write("{:2d},".format(skipped_last) + line)
                        num_lines += 1
                        skipped_last = False
                    else:
                        print(f"discarding: ", line, end='')
                        print(f"  line.count(',') == num_fields  {line.count(",") == num_fields}   \
\nAND num_text == num_text_run {num_text == num_text_run} \
\nAND re.search(r'[^a-zA-Z0-9+-_.:, ]', line[:-1]) is None {re.search(r'[^a-zA-Z0-9+-_.:, ]', line[:-1]) is None} \
\nAND (num_lines == 0 or ((num_lines+1) % skip) == 0) {(num_lines == 0 or ((num_lines+1) % skip) == 0)} \
\nAND line.count(comment_str) == 0 {line.count(comment_str) == 0}")
                        print(f"{line.count(',')=} {num_fields=}")
                        print(f"{line[-1]=}")
                        print(f"{num_text=} {num_text_run=}")
                        num_skips += 1
                        skipped_last = True
                    num_lines += 1
    if not num_lines:
        csv_file = None
        print("I(write_clean_file): no data to write")
        if not unit_key_found:
            print("W(write_clean_file):  unit_key not found in ", basename, ".  Looking with '{:s}'".format(unit_key))
    else:
        print("Wrote(write_clean_file):", csv_file, num_lines, "lines", num_skips, "skips", num_fields, "fields")
    return csv_file
