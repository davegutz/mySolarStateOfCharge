#! /bin/sh
# noinspection PySingleQuotedDocstring
"exec" "`dirname $0`/venv/bin/python3" "$0" "$@"
from PlotKiller import show_killer

#  #! /Users/daveg/Documents/GitHub/mySolarStateOfCharge/SOC_Particle/py/venv/bin/python
# The #! operates for macOS only. 'Python Launcher' (Python Script Preferences) option for 'Allow override with #! in script' is checked.
#  Graphical interface to Test State of Charge application
#  Run in PyCharm
#     or
#  python3 GUI_TestSOC.py
#
#  2023-Jun-15  Dave Gutz   Create
# Copyright (C) 2026 Dave Gutz
#
# noinspection PyTypeChecker,PyArgumentList,PyCallingNonCallable,PyUnfilledParameters,SpellCheckingInspection,PyPep8Naming,PyUnboundLocalVariable,PyShadowingNames,PyShadowingBuiltins
# type: ignore
# pylint: disable=all, invalid-name, used-before-assignment, redefined-outer-name, redefined-builtin
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

"""Define a class to manage configuration using files for memory (poor man's database)"""
import sys
import os
from pathlib import Path, PurePosixPath
import time
from configparser import ConfigParser
import re
from tkinter import filedialog
import tkinter.simpledialog
import tkinter.messagebox
from CompareHistHist import compare_hist_hist
from CompareHistSim import compare_hist_sim
from CompareRunSim import compare_run_sim
from CompareRunRun import compare_run_run
from CompareRunHist import compare_run_hist
from CountdownTimer import CountdownTimer
import matplotlib.pyplot as plt
import shutil
import pyperclip
import shlex
import subprocess
import datetime
import platform
from Colors import Colors
from test_soc_util import run_shell_cmd
if platform.system() == 'Darwin':
    # noinspection PyUnresolvedReferences
    from ttwidgets import TTButton as myButton  # Need this for  macOS - ignore warnings
else:
    from tkinter import Button as myButton
bg_color = 'lightgray'
if sys.version_info.major == 3 and sys.version_info.minor < 12:
    # noinspection PyUnusedImports
    import pyautogui
else:
    try:
        from evdev import UInput, ecodes as ev
        _kb_backend = 'evdev'
    except ImportError:
        from pynput.keyboard import Key, Controller
        _kb_backend = 'pynput'
from GUI_common import (
    _Tee,
    add_to_clip_board,
    battery_list,
    Begini,
    contain_all,
    copy_clean,
    create_file_key,
    create_file_txt,
    default_auto,
    default_auto_header,
    default_dict,
    empty_file,
    ExRoot,
    lookup,
    macro_lookup,
    macro_sel_list,
    no_shift_soc_s,
    plat,
    plink_connection,
    register_last_task,
    sel_list,
    sel_list1,
    size_of,
    unit_list,
)

sys.stdout.write("\033]0;SOC\007")
sys.stdout.flush()

# Tee stdout/stderr to a log file so Console.app shows output when launched as a .app bundle
_log_dir = os.path.expanduser("~/Library/Logs") if plat == 'darwin' else os.path.expanduser("~")
os.makedirs(_log_dir, exist_ok=True)
_log_file = open(os.path.join(_log_dir, "GUI_TestSOC.log"), 'a', buffering=1)

plink_pid = None
linux_terminal_pid = None  # Linux: terminal process (xterm/qterminal) — killed explicitly on stop
cmd_window_pid = None  # Windows: cmd.exe /k window that hosts plink — killed separately on stop
cmd_window_title_prefix = 'plink-terminal-server'  # Unique title set in bat file; used as taskkill fallback
cmd_window_pids = set()  # Windows: every cmd.exe window we ever spawned this session (belt+suspenders)
run_start_time = None  # Set at grab_start, used to print elapsed time on DONE
auto_running = False  # Track if AUTO process is active
auto_fig_list = None  # Handles to figures from the most recently completed AUTO case
auto_case_index = 0   # Current AUTO case index (0-based)
auto_case_total = 0   # Total number of AUTO cases
_monitor_after_id = None  # Pending after() ID for monitor_plink_done; used to cancel stale loops



sys.stdout = _Tee(sys.__stdout__, _log_file)
sys.stderr = _Tee(sys.__stderr__, _log_file)

# Executive class to control the global variables
class Exec:
    def __init__(self, cf_=None, ind=None, level=None, path_disp_len_=25):
        self.root_config = None
        self.cf = cf_
        self.ind = ind
        self.level = level
        self.path_disp_len = path_disp_len_
        self.script_loc = Path(__file__).resolve().parent.as_posix()
        self.config_path = str(PurePosixPath(self.script_loc) / 'root_config.ini')
        # self.root_config = None
        self.load_root_config(self.config_path)
        self.dataReduction_folder = self.cf[self.ind]['dataReduction_folder']
        self.version = self.cf[self.ind]['version']
        self.battery = self.cf[self.ind]['battery']
        self.unit = self.cf[self.ind]['unit']
        if self.version is None:
            self.version = 'undefined'
        self.version_path = str(PurePosixPath(self.dataReduction_folder or '.') / (self.version or 'undefined'))
        try:
            os.makedirs(self.version_path, exist_ok=True)
        except OSError:
            tk.messagebox.showerror(title="Error",
                                    message=self.version_path + " unavailable. Abort opening\nTurn on Drive & refresh" +
                                                                " dataReduction Folder.")
        # Following need explicit shallow copy lines
        self.folder_button = myButton(master, text=self.dataReduction_folder[-20:],
                                      command=self.enter_data_reduction_folder, fg="blue", bg=bg_color)
        self.version_button = None
        self.unit_button = None
        self.battery_button = None
        self.key_label = None
        self.file_txt = None
        self.file_path = None
        self.file_exists = None
        self.dataReduction_folder_exists = None
        self.key_exists_in_file = None
        self.label = None
        self.key = None

    def __copy__(self):
        """Shallow copy function"""
        instance = object.__new__(Exec)
        vars(instance).update(vars(self))
        return instance

    def create_file_path_and_key(self, name_override=None):
        if name_override is None:
            self.file_txt = create_file_txt(self.cf['others']['option'], self.unit, self.battery)
            self.key = create_file_key(self.version, self.unit, self.battery)
            print('version', self.version, 'key', self.key)
        else:
            self.file_txt = create_file_txt(name_override, self.unit, self.battery)
            self.key = create_file_key(self.version, self.unit, self.battery)
        self.file_path = str(PurePosixPath(self.version_path or '.') / (self.file_txt or 'undefined'))
        self.update_file_label()
        self.file_exists = Path(self.file_path).is_file()
        self.update_file_label()
        self.update_key_label()
        self.update_folder_button()

    def enter_battery(self):
        answer = tk.simpledialog.askstring(title=self.level,
                                           prompt="Enter battery e.g. 'bb for Battleborn', 'ch' or 'chg' for CHINS:")
        if answer is None or answer == () or answer == '':
            print('enter operation cancelled')
            return
        self.battery = answer
        self.cf[self.ind]['battery'] = self.battery
        self.cf.save_to_file()
        self.battery_button.config(text=self.battery)
        self.create_file_path_and_key()
        self.update_key_label()

    def enter_data_reduction_folder(self):
        answer = tk.filedialog.askdirectory(title="Select a destination (i.e. Library) dataReduction folder",
                                            initialdir=self.dataReduction_folder)
        if answer is None or answer == () or answer == '' or answer == '':
            print('enter operation cancelled')
            return
        self.dataReduction_folder = answer
        self.cf[self.ind]['dataReduction_folder'] = self.dataReduction_folder
        self.cf.save_to_file()
        self.folder_button.config(text=self.dataReduction_folder[self.path_disp_len:])
        self.update_folder_button()

    def enter_unit(self):
        answer = tk.simpledialog.askstring(title=self.level, initialvalue=self.unit,
                                           prompt="Enter unit e.g. 'pro0p', 'pro1a', 'pro2p2'"
                                                  "'pro2p2_hi_lo', 'pro3p2', 'pro3p2_hi_lo', 'pro4p2', 'soc0p', 'soc1a',"
                                                  "'soc2p2_hi_lo', 'soc3p2_hi_lo', 'soc4p2_hi_lo':")
        if answer is None or answer == () or answer == '':
            print('enter operation cancelled')
            return
        self.unit = answer
        self.cf[self.ind]['unit'] = self.unit
        self.cf.save_to_file()
        self.unit_button.config(text=self.unit)
        self.create_file_path_and_key()
        self.update_key_label()
        self.update_file_label()

    def enter_version(self):
        answer = tk.simpledialog.askstring(title=__file__, prompt="Enter version <vYYYYMMDD>:",
                                           initialvalue=self.version)
        if answer is None or answer == () or answer == '':
            print('enter operation cancelled')
            return
        self.version = answer
        self.cf[self.ind]['version'] = self.version
        self.cf.save_to_file()
        self.version_button.config(text=self.version)
        self.version_path = str(PurePosixPath(self.dataReduction_folder or '.') / (self.version or 'undefined'))
        os.makedirs(self.version_path, exist_ok=True)
        self.create_file_path_and_key()
        self.update_key_label()
        self.label.config(text=self.file_txt)

    def load_root_config(self, config_file_path):
        self.root_config = ConfigParser()
        if Path(config_file_path).is_file():
            self.root_config.read(config_file_path)
        else:
            with open(config_file_path, 'w') as cfg_file:
                self.root_config.add_section('Root Preferences')
                rec_folder_path = str(Path.home() / 'Documents' / 'Recordings')
                if not Path(rec_folder_path).exists():
                    os.makedirs(rec_folder_path)
                self.root_config.set('Root Preferences', 'recordings path', rec_folder_path)
                self.root_config.write(cfg_file)
        return self.root_config

    def save_root_config(self, config_path_):
        if Path(config_path_).is_file():
            with open(config_path_, 'w') as cfg_file:
                self.root_config.write(cfg_file)
            print('Saved', config_path_)
        return self.root_config

    def super_shallow_copy(self, other):
        self.level = other.level
        self.path_disp_len = other.path_disp_len
        self.script_loc = other.script_loc
        self.config_path = other.config_path
        self.root_config = other.root_config
        self.dataReduction_folder = other.dataReduction_folder
        self.version = other.version
        self.battery = other.battery
        self.unit = other.unit
        self.version_path = other.version_path
        self.file_txt = other.file_txt
        self.file_path = other.file_path
        self.file_exists = other.file_exists
        self.dataReduction_folder_exists = other.dataReduction_folder_exists
        self.key_exists_in_file = other.key_exists_in_file
        self.key = other.key

    def update_battery_stuff(self):
        self.cf[self.ind]['version'] = self.version
        self.cf[self.ind]['unit'] = self.unit
        self.cf[self.ind]['battery'] = self.battery
        self.cf[self.ind]['dataReduction_folder'] = self.dataReduction_folder
        self.cf.save_to_file()
        self.create_file_path_and_key()
        self.update_folder_button()
        self.update_version_button()
        self.update_unit_button()
        self.update_battery_button()
        self.update_key_label()
        self.update_file_label()

    def update_file_label(self):
        self.label.config(text=self.file_txt)
        if self.file_exists:
            self.label.config(bg='lightgreen')
        else:
            self.label.config(bg='pink')

    def update_battery_button(self):
        self.battery_button.config(text=self.battery)

    def update_folder_button(self):
        if Path(self.dataReduction_folder).exists():
            self.dataReduction_folder_exists = True
        else:
            self.dataReduction_folder_exists = False
        self.folder_button.config(text=self.dataReduction_folder[-self.path_disp_len:])
        if self.dataReduction_folder_exists:
            self.folder_button.config(bg='lightgreen')
        else:
            self.folder_button.config(bg='pink')

    def update_key_label(self):
        self.key_label.config(text=self.key)
        self.key_exists_in_file = False
        if Path(self.file_path).is_file():
            for line in open(self.file_path, 'r'):
                if re.search(self.key, line):
                    self.key_exists_in_file = True
                    break
        if self.key_exists_in_file:
            self.key_label.config(bg='lightgreen')
        else:
            self.key_label.config(bg='pink')
        test_filename.set(plink_connection.get(Test.unit or '', ''))

    def update_unit_button(self):
        self.unit_button.config(text=self.unit)

    def update_version_button(self):
        self.version_button.config(text=self.version)


# Compare run driver
def clear_data_silent(nowait=True):
    clear_data(silent=True, nowait=nowait)


def clear_data_verbose():
    clear_data(silent=False)


def clear_data(silent=False, nowait=False):
    if Path(plink_test_csv_path.get()).is_file():
        enter_size = plink_size()  # bytes
        time.sleep(1.)
        wait_size = plink_size()  # bytes
    else:
        enter_size = 0
        wait_size = 0
    if enter_size > 64:  # bytes
        if wait_size > enter_size and not nowait:
            if not silent:
                print('stop data first')
            tkinter.messagebox.showwarning(message="stop data first")
        else:
            # create empty file
            if not save_plink():
                if not silent:
                    tkinter.messagebox.showwarning(message="plink may be open already")
                else:
                    update_data_buttons()
    else:
        if not silent:
            print('plink test file non-existent or too small (<64 bytes) probably already done')
            tkinter.messagebox.showwarning(message="Nothing to clear")


# Choose file to perform compare_hist_hist on
def compare_hist_hist_choose():
    # Select file
    print('compare_hist_hist_choose')
    testpaths = filedialog.askopenfilenames(title='Choose test file(s)', filetypes=[('csv', '.csv')],
                                            initialdir=Test.dataReduction_folder)
    if testpaths is None or testpaths == '':
        print("No file chosen")
    else:
        for testpath in testpaths:
            test_folder_path, test_parent, test_basename, test_txt, test_key = contain_all(testpath)
            if test_key != '':
                run_path = filedialog.askopenfilename(title='Choose reference file', filetypes=[('csv', '.csv')],
                                                      initialdir=Ref.dataReduction_folder)
                run_folder_path, ref_parent, ref_basename, ref_txt, ref_key = contain_all(run_path)
                print('GUI_PlinkSOC compare_hist_hist_choose:  Ref', ref_basename, ref_key)
                print('GUI_PlinkSOC compare_hist_hist_choose:  Test', test_basename, test_key)
                compare_hist_hist(data_file_run=run_path, unit_key_run=ref_key,
                                  data_file_tst=testpath, unit_key_tst=test_key,
                                  dt_resample=30.,
                                  terse=terse.get())
            else:
                tk.messagebox.showerror(message='key not found in' + testpath)
        update_data_buttons()


# Choose file to perform compare_run_sim on
def compare_hist_sim_choose():
    # Select file
    print('compare_hist_sim_choose')
    testpaths = filedialog.askopenfilenames(title='Please select files', filetypes=[('csv', '.csv')],
                                            initialdir=Test.dataReduction_folder)
    if testpaths is None or testpaths == '':
        print("No file chosen")
    else:
        update_data_buttons()
        for testpath in testpaths:
            test_folder_path, test_parent, basename, test_txt, key = contain_all(testpath)
            if key != '':
                answer = tk.simpledialog.askinteger(title=__file__,
                                                    prompt="Simulation re-construction sample time in seconds",
                                                    initialvalue=900)
                if answer is None:
                    print('enter operation cancelled')
                    return
                compare_hist_sim(data_file=testpath, unit_key=key, dt_resample=answer, terse=terse.get(),
                                 strict_overplot=strict_overplot.get())
            else:
                tk.messagebox.showerror(message='key not found in' + testpath)


def compare_hist_to_sim():
    register_last_task(compare_hist_to_sim)
    if modeling.get():
        update_data_buttons()
        print('compare_hist_to_sim.  save_pdf_path', str(PurePosixPath(Test.version_path) / 'figures'))
        answer = tk.simpledialog.askinteger(title=__file__, prompt="Simulation re-construction sample time in seconds",
                                            initialvalue=10)
        if answer is None:
            print('enter operation cancelled')
            return
        compare_hist_sim(data_file=Test.file_path, unit_key=Test.key, use_mon_csv=True, dt_resample=answer,
                         terse=terse.get(), strict_overplot=strict_overplot.get(), hardcopy=hardcopy.get())
    else:
        print('not possible')


def compare_run(show_killer_=True):
    register_last_task(compare_run)
    if not Test.key_exists_in_file:
        tkinter.messagebox.showwarning(message="Test Key '" + Test.key + "' does not exist in " + Test.file_txt)
        return
    update_data_buttons()
    if modeling.get():
        print('compare_run_sim.  save_pdf_path', str(PurePosixPath(Test.version_path) / 'figures'))
        return compare_run_sim(data_file=Test.file_path, unit_key=Test.key, strict_overplot=strict_overplot.get(),
                               terse=terse.get(), hardcopy=hardcopy.get(), show_killer_=show_killer_,
                               shift_soc_s=option.get() not in no_shift_soc_s)
    else:
        if not Ref.key_exists_in_file:
            tkinter.messagebox.showwarning(message="Ref Key '" + Ref.key + "' does not exist in " + Ref.file_txt)
            return
        print('GUI_TestSOC compare_run:  Ref', Ref.file_path, Ref.key)
        print('GUI_TestSOC compare_run:  Test', Test.file_path, Test.key)
        keys = [(Ref.file_txt, Ref.key), (Test.file_txt, Test.key)]
        return compare_run_run(keys=keys, data_file_folder_run=Ref.version_path, data_file_folder_test=Test.version_path,
                               terse=terse.get(), hardcopy=hardcopy.get())



def compare_run_to_hist():
    register_last_task(compare_run_to_hist)
    if not Test.key_exists_in_file:
        tkinter.messagebox.showwarning(message="Test Key '" + Test.key + "' does not exist in " + Test.file_txt)
        return
    update_data_buttons()
    if modeling.get():
        print('compare_hist_to_sim.  save_pdf_path', str(PurePosixPath(Test.version_path) / 'figures'))
        compare_run_hist(data_file=Test.file_path, unit_key=Test.key, strict_overplot=strict_overplot.get(),
                        terse=terse.get())
    else:
        print('not possible')


def compare_hist_hist_run():
    register_last_task(compare_hist_hist_run)
    if not Test.key_exists_in_file:
        tkinter.messagebox.showwarning(message="Test Key '" + Test.key + "' does not exist in " + Test.file_txt)
        return
    if not Ref.key_exists_in_file:
        tkinter.messagebox.showwarning(message="Ref Key '" + Ref.key + "' does not exist in " + Ref.file_txt)
        return
    update_data_buttons()
    answer = tk.simpledialog.askinteger(title=__file__, prompt="Simulation re-construction sample time in seconds",
                                        initialvalue=10)
    if answer is None:
        print('enter operation cancelled')
        return
    print('GUI_TestSOC compare_hist_hist_run:  Ref', Ref.file_path, Ref.key)
    print('GUI_TestSOC compare_hist_hist_run:  Test', Test.file_path, Test.key)
    compare_hist_hist(data_file_run=Ref.file_path, unit_key_run=Ref.key,
                      data_file_tst=Test.file_path, unit_key_tst=Test.key,
                      dt_resample=answer, terse=terse.get(), hardcopy=hardcopy.get())


# Choose file to perform compare_run_run on
def compare_run_run_choose():
    # Select file
    print('compare_run_run_choose')
    testpaths = filedialog.askopenfilenames(title='Choose test file(s)', filetypes=[('csv', '.csv')],
                                            initialdir=Test.dataReduction_folder)
    if testpaths is None or testpaths == '':
        print("No file chosen")
    else:
        for testpath in testpaths:
            test_folder_path, test_parent, test_basename, test_txt, test_key = contain_all(testpath)
            if test_key != '':
                ref_path = filedialog.askopenfilename(title='Choose reference file', filetypes=[('csv', '.csv')],
                                                      initialdir=Ref.dataReduction_folder)
                ref_folder_path, ref_parent, ref_basename, ref_txt, ref_key = contain_all(ref_path)
                print('GUI_TestSOC compare_run_run_choose:  Ref', ref_basename, ref_key)
                print('GUI_TestSOC compare_run_run_choose:  Test', test_basename, test_key)
                keys = [(ref_basename, ref_key), (test_basename, test_key)]
                compare_run_run(keys=keys, data_file_folder_run=ref_folder_path, data_file_folder_test=test_folder_path,
                                sync_to_ctime=True)
            else:
                tk.messagebox.showerror(message='key not found in' + testpath)
        update_data_buttons()


# Choose file to perform compare_run_sim on
def compare_run_sim_choose(show_killer_=True):
    # Select file
    print('compare_run_sim_choose')
    testpaths = filedialog.askopenfilenames(title='Please select files', filetypes=[('csv', '.csv')],
                                            initialdir=Test.dataReduction_folder)
    if testpaths is None or testpaths == '':
        print("No file chosen")
    else:
        for testpath in testpaths:
            test_folder_path, test_parent, basename, test_txt, key = contain_all(testpath)
            if key != '':
                compare_run_sim(data_file=testpath, unit_key=key, strict_overplot=strict_overplot.get(),
                        terse=terse.get(), show_killer_=show_killer_)
            else:
                tk.messagebox.showerror(message='key not found in' + testpath)
        update_data_buttons()


def compare_run_ver_batch():
    from CompareRunVer import temp_folder, find_pairs, compare_pair, report, plot_diffs, show_batch_diffs

    plink_path = Path(plink_test_csv_path.get())
    auto_plink_path = plink_path.parent / 'auto_plink.csv'
    if not auto_plink_path.is_file():
        tkinter.messagebox.showerror(title="File Not Found",
                                     message=f"Could not find {auto_plink_path}")
        return

    try:
        with open(auto_plink_path, 'r') as f:
            lines = f.readlines()

        header_fields = []
        data_rows = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                if not header_fields:
                    header_fields = [f.strip() for f in line.lstrip('#').split(',') if f.strip()]
                continue
            if header_fields:
                values = [v.strip() for v in line.split(',')]
                if len(values) >= len(header_fields):
                    data_rows.append(dict(zip(header_fields, values)))

        if not data_rows:
            tkinter.messagebox.showwarning(message="No valid data rows in auto_plink.csv")
            return

        tol = 1e-3
        all_fig_list = []
        all_fig_files = []
        last_pdf_path = None
        problem_cases = []  # list of (desc, reason)

        for config in data_rows:
            version = config.get('version', Test.version)
            macro_val = config.get('macro', '')
            option_val = macro_val if macro_val in lookup else ''
            case_desc = f"version={version!r}, macro={macro_val!r}"

            temp_dir = temp_folder(version)
            if not Path(temp_dir).is_dir():
                reason = f"temp folder not found: {temp_dir}"
                print(f"\033[91mRunVer SKIP  {case_desc}: {reason}\033[0m")
                problem_cases.append((case_desc, reason))
                continue

            pairs = find_pairs(temp_dir, option=option_val)
            pairs = [(r, v) for r, v in pairs if '_mon_run' in r.name or '_sim_run' in r.name]
            if not pairs:
                reason = f"no mon/sim pairs (option={option_val!r})"
                print(f"\033[91mRunVer SKIP  {case_desc}: {reason}\033[0m")
                problem_cases.append((case_desc, reason))
                continue

            try:
                results = [compare_pair(run, ver, tol) for run, ver in pairs]
                report(results, tol, option=option_val, macro=macro_val)
                ret = plot_diffs(results, data_file=Test.file_path, show_killer_=False)
                if ret is not None:
                    figs, files, pdf_path = ret
                    all_fig_list.extend(figs)
                    all_fig_files.extend(files)
                    if pdf_path:
                        last_pdf_path = pdf_path
            except Exception as case_e:
                reason = str(case_e)
                print(f"\033[91mRunVer FAIL  {case_desc}: {reason}\033[0m")
                problem_cases.append((case_desc, reason))

        n_total = len(data_rows)
        n_problems = len(problem_cases)
        if not problem_cases:
            tkinter.messagebox.showinfo(title="RunVer Complete",
                                        message=f"All {n_total} case(s) executed successfully.")
        else:
            print("\033[91m--- RunVer problem summary ---\033[0m")
            for desc, reason in problem_cases:
                print(f"\033[91m  {desc}: {reason}\033[0m")
            tkinter.messagebox.showwarning(
                title="RunVer Complete",
                message=f"{n_total - n_problems} of {n_total} case(s) ran.\n"
                        f"{n_problems} case(s) had problems — check the status output for details.")

        show_batch_diffs(all_fig_list, all_fig_files, last_pdf_path)

    except Exception as e:
        print(f"compare_run_ver_batch: {e}")
        tkinter.messagebox.showerror(title="Error", message=str(e))


def run_sim_all_batch():
    plink_path = Path(plink_test_csv_path.get())
    auto_plink_path = plink_path.parent / 'auto_plink.csv'
    if not auto_plink_path.is_file():
        tkinter.messagebox.showerror(title="File Not Found",
                                     message=f"Could not find {auto_plink_path}")
        return

    try:
        with open(auto_plink_path, 'r') as f:
            lines = f.readlines()

        header_fields = []
        data_rows = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                if not header_fields:
                    header_fields = [f.strip() for f in line.lstrip('#').split(',') if f.strip()]
                continue
            if header_fields:
                values = [v.strip() for v in line.split(',')]
                if len(values) >= len(header_fields):
                    data_rows.append(dict(zip(header_fields, values)))

        if not data_rows:
            tkinter.messagebox.showwarning(message="No valid data rows in auto_plink.csv")
            return

        all_fig_list = []
        problem_cases = []  # list of (desc, reason)

        for config in data_rows:
            folder = config.get('folder', Test.dataReduction_folder)
            version = config.get('version', Test.version)
            battery = config.get('battery', Test.battery)
            macro = config.get('macro', '')
            case_desc = f"version={version!r}, macro={macro!r}"

            version_path = str(PurePosixPath(folder) / version)
            file_txt = create_file_txt(macro, Test.unit, battery)
            file_path = str(PurePosixPath(version_path) / file_txt)
            key = create_file_key(version, Test.unit, battery)

            if not Path(file_path).is_file():
                reason = f"file not found: {file_path}"
                print(f"\033[91mRunSimAll SKIP  {case_desc}: {reason}\033[0m")
                problem_cases.append((case_desc, reason))
                continue

            try:
                print(f"run_sim_all_batch: {file_path}")
                result = compare_run_sim(data_file=file_path, unit_key=key,
                                         strict_overplot=True, terse=True, hardcopy=True,
                                         show_killer_=False, fig_list=all_fig_list,
                                         shift_soc_s=macro not in no_shift_soc_s)
                if result is not None:
                    all_fig_list = result[0]
            except Exception as case_e:
                reason = str(case_e)
                print(f"\033[91mRunSimAll FAIL  {case_desc}: {reason}\033[0m")
                problem_cases.append((case_desc, reason))

        n_total = len(data_rows)
        n_problems = len(problem_cases)
        if not problem_cases:
            tkinter.messagebox.showinfo(title="RunSimAll Complete",
                                        message=f"All {n_total} case(s) executed successfully.")
        else:
            print("\033[91m--- RunSimAll problem summary ---\033[0m")
            for desc, reason in problem_cases:
                print(f"\033[91m  {desc}: {reason}\033[0m")
            tkinter.messagebox.showwarning(
                title="RunSimAll Complete",
                message=f"{n_total - n_problems} of {n_total} case(s) ran.\n"
                        f"{n_problems} case(s) had problems — check the status output for details.")

        if all_fig_list:
            string = 'plots ' + str(all_fig_list[0].number) + ' - ' + str(all_fig_list[-1].number)
            show_killer(string, 'RunSimAll', fig_list=all_fig_list, hardcopy=False)
        else:
            print("run_sim_all_batch: no figures produced")

    except Exception as e:
        print(f"run_sim_all_batch: {e}")
        tkinter.messagebox.showerror(title="Error", message=str(e))


def enter_mod_in_app():
    answer = tk.simpledialog.askinteger(title=__file__, prompt="enter the value of Modeling in app to assume", initialvalue=mod_in_app.get())
    if answer is None:
        print('enter operation cancelled')
        return
    mod_in_app.set(answer)
    cf['others']['mod_in_app'] = str(mod_in_app.get())
    cf.save_to_file()
    mod_in_app_button.config(text=mod_in_app.get())


def grab_macro():
    register_last_task(grab_macro)
    add_to_clip_board(macro.get())
    macro_button.config(bg='yellow', activebackground='yellow', fg='black', activeforeground='black')
    init_button.config(bg=bg_color, activebackground=bg_color, fg='black', activeforeground='black')
    start_button.config(bg='black', activebackground='black', fg='#00ff00', activeforeground='#00ff00')
    get_time_button.config(bg=bg_color, activebackground=bg_color, fg='black', activeforeground='purple')


def grab_init(command_to_append='', force_if_ready=False, force_kill=False, fg_color='#ffffff'):
    register_last_task(grab_init)
    # Grab command to update time in EEPROM
    try:
        current_ut = 'UT' + str(int(time.time())) + ';'
        print(f"current_ut {current_ut}")
    except AttributeError:
        current_ut = ''
        print(f"current_ut blank ***No Internet??")
    init_command = init.get() + current_ut
    if command_to_append:
        init_command += command_to_append
    print(f"Init command to paste: {init_command}")
    add_to_clip_board(init_command)
    # Grab the rest
    grab_all_nominal()
    init_button.config(bg='yellow', activebackground='yellow', fg='black', activeforeground='black')
    if not force_if_ready:
        clear_data_silent()
        print('cleared plink data file')
    else:
        print('skipping clear_data because force_if_ready is True')
    Test.create_file_path_and_key()
    Test.update_key_label()
    return start_plink(command_to_paste=init_command, force_if_ready=force_if_ready, force_kill=force_kill, fg_color=fg_color)


def open_plink_window():
    register_last_task(open_plink_window)
    grab_all_nominal()
    Test.create_file_path_and_key()
    Test.update_key_label()
    start_plink(fg_color='#F5DEB3', bg_color='#2F4F4F')


def _cancel_monitor_plink():
    global _monitor_after_id
    if _monitor_after_id is not None:
        try:
            master.after_cancel(_monitor_after_id)
        except Exception:
            pass
        _monitor_after_id = None


def monitor_plink_done():
    global _monitor_after_id
    _monitor_after_id = None
    done_detected = False
    if Path(plink_test_csv_path.get()).is_file():
        try:
            with open(plink_test_csv_path.get(), 'rb') as f:
                f.seek(0, 2)
                size = f.tell()
                # Read last 1024 bytes to check for ***DONE***
                f.seek(max(0, size - 1024))
                last_data = f.read().decode('utf-8', errors='ignore')
                if '***DONE***' in last_data:
                    elapsed = time.time() - run_start_time if run_start_time is not None else float('nan')
                    print(f"***DONE*** detected in {plink_test_csv_path.get()}  elapsed={elapsed:.1f}s")
                    if auto_running:
                        return  # AUTO's check_completion owns this path
                    done_detected = True
        except Exception as e:
            print(f"Error monitoring plink file: {e}")
    if done_detected:
        save_data()
        tk.messagebox.showinfo(title='Done ' + start_button.cget('text'), message='Run Complete')
        return
    _monitor_after_id = master.after(1000, monitor_plink_done)


def grab_start():
    global run_start_time
    run_start_time = time.time()
    register_last_task(grab_start)
    start_command = start.get()
    print(f"Start command to paste: {start_command}")
    # Force restart if the already running plink process is 'READY'
    if look_plink(platform.system()):
        if not is_plink_ready():
            print("Plink is already open but NOT READY.")
            tkinter.messagebox.showinfo(title="Not Ready",
                                       message="Please wait until terminal is READY or run the START HERE button.")
            return

        # If it is READY, we restart and only send the start_command
        # We still need to call grab_all_nominal and update labels
        grab_all_nominal()
        Test.create_file_path_and_key()
        Test.update_key_label()
        if not start_plink(command_to_paste=start_command, force_if_ready=True):
            return
    else:
        # If not open at all, use grab_init to bundle both init and start
        if not grab_init(command_to_append=start_command, force_if_ready=True, fg_color='#00ff00'):
            return

    add_to_clip_board(start_command)
    grab_all_nominal()
    save_data_button.config(bg=bg_color, activebackground=bg_color, fg='black', activeforeground='black',
                            text='save data')
    save_data_as_button.config(bg=bg_color, activebackground=bg_color, fg='black', activeforeground='black',
                               text='save data as')
    grab_all_nominal()
    start_button.config(bg='yellow', activebackground='yellow', fg='black', activeforeground='black')
    start_timer()
    _cancel_monitor_plink()
    empty_file(plink_test_csv_path.get())  # clear stale DONE before monitoring new run
    monitor_plink_done()


def grab_all_nominal():
    macro_button.config(bg=bg_color, activebackground=bg_color, fg='black', activeforeground='black')
    init_button.config(bg=bg_color, activebackground=bg_color, fg='black', activeforeground='purple')
    start_button.config(bg='black', activebackground='black', fg='#00ff00', activeforeground='#00ff00')
    get_time_button.config(bg=bg_color, activebackground=bg_color, fg='black', activeforeground='black')


def grab_time():
    current_ut = 'UT' + str(int(time.time())) + ';'
    add_to_clip_board(current_ut)
    grab_all_nominal()
    get_time_button.config(bg='yellow', activebackground='yellow', fg='black', activeforeground='black',
                           text=current_ut)
    print('UT in paste buffer')


def handle_modeling(*_args):
    cf['others']['modeling'] = str(modeling.get())
    cf.save_to_file()
    if modeling.get():
        ref_remove()
    else:
        ref_restore()


def handle_macro(*_args):
    lookup_macro()
    macro_option_ = macro_option.get()

    # Check if this is what you want to do (skipped in AUTO)
    if not auto_running:
        if macro_option_.__contains__('CH'):
            if Test.battery == 'bb' or Ref.battery == 'bb':
                confirmation = tk.messagebox.askyesno('query sensical', 'Test/Ref are "bb." Continue?')
                if not confirmation:
                    print('start over')
                    tkinter.messagebox.showwarning(message='try again')
                    option.set('try again')
                    return
        elif macro_option_.__contains__('BB'):
            if Test.battery == 'ch' or Ref.battery == 'ch' or Test.battery == 'chg' or Ref.battery == 'chg':
                confirmation = tk.messagebox.askyesno('query sensical', 'Test/Ref are "ch." Continue?')
                if not confirmation:
                    print('start over')
                    tkinter.messagebox.showwarning(message='try again')
                    option.set('try again')
                    return

    macro_option_show.set(macro_option_)
    cf['others']['macro'] = macro_option_
    cf.save_to_file()
    macro_button.config(bg=bg_color, activebackground=bg_color, fg='black', activeforeground='purple')


def handle_option(*_args):
    lookup_start()
    option_ = option.get()

    # Check if this is what you want to do (skipped in AUTO)
    if not auto_running:
        if option_.__contains__('CH'):
            if Test.battery == 'bb' or Ref.battery == 'bb':
                confirmation = tk.messagebox.askyesno('query sensical', 'Test/Ref are "bb." Continue?')
                if not confirmation:
                    print('start over')
                    tkinter.messagebox.showwarning(message='try again')
                    option.set('try again')
                    return
        elif option_.__contains__('BB'):
            if Test.battery == 'ch' or Ref.battery == 'ch' or Test.battery == 'chg' or Ref.battery == 'chg':
                confirmation = tk.messagebox.askyesno('query sensical', 'Test/Ref are "cc." Continue?')
                if not confirmation:
                    print('start over')
                    tkinter.messagebox.showwarning(message='try again')
                    option.set('try again')
                    return

    option_show.set(option_)
    cf['others']['option'] = option_
    cf.save_to_file()
    Test.create_file_path_and_key()
    Ref.create_file_path_and_key()
    save_data_button.config(bg=bg_color, activebackground=bg_color, fg='black', activeforeground='black',
                            text='save data')
    save_data_as_button.config(bg=bg_color, activebackground=bg_color, fg='black', activeforeground='black',
                               text='save data as')
    start_button.config(bg='black', activebackground='black', fg='#00ff00', activeforeground='#00ff00')
    update_data_buttons()


def handle_run_battery(*_args):
    Ref.battery = ref_battery.get()
    Ref.update_battery_stuff()
    update_data_buttons()


def handle_run_unit(*_args):
    Ref.unit = ref_unit.get()
    Ref.update_battery_stuff()
    update_data_buttons()


def handle_strict_overplot(*_args):
    cf['others']['strict_overplot'] = str(strict_overplot.get())
    cf.save_to_file()


def handle_hardcopy(*_args):
    cf['others']['hardcopy'] = str(hardcopy.get())
    cf.save_to_file()


def handle_terse(*_args):
    cf['others']['terse'] = str(terse.get())
    cf.save_to_file()


def handle_auto_overwrite(*_args):
    cf['others']['auto_overwrite'] = str(auto_overwrite.get())
    cf.save_to_file()



def handle_test_battery(*_args):
    Test.battery = test_battery.get()
    Test.update_battery_stuff()
    update_data_buttons()


def handle_test_unit(*_args):
    Test.unit = test_unit.get()
    Test.update_battery_stuff()
    update_data_buttons()


def _close_all_plink_windows_windows(silent=True):
    """Close every cmd.exe window we ever opened in this session, plus any matching title."""
    global cmd_window_pid, cmd_window_pids
    for _pid in list(cmd_window_pids):
        try:
            run_shell_cmd(f'taskkill /f /pid {_pid}', silent=silent)
        except Exception as e:
            print(f"Error closing cmd window PID {_pid}: {e}")
    cmd_window_pids.clear()
    cmd_window_pid = None
    # Belt-and-suspenders fallback: close any leftover window matching our title prefix.
    # Wildcard form requires no /im — taskkill matches by window title filter.
    try:
        run_shell_cmd(f'taskkill /f /fi "WINDOWTITLE eq {cmd_window_title_prefix}*"', silent=silent)
    except Exception as e:
        print(f"Title-based taskkill fallback failed: {e}")


def _pid_comm(pid):
    try:
        return subprocess.check_output(['ps', '-o', 'comm=', '-p', str(pid)]).decode().strip()
    except Exception:
        return ''


def _pid_ppid(pid):
    try:
        out = subprocess.check_output(['ps', '-o', 'ppid=', '-p', str(pid)]).decode().strip()
        return int(out) if out else None
    except Exception:
        return None


def _is_self_or_ancestor(pid):
    # Under an IDE wrapper (PyCharm) the GUI's Python process — or one of its ancestors —
    # can become a subreaper, so plink's reported parent ends up being us. Refuse to kill that.
    if pid is None or pid <= 1:
        return True
    cur = os.getpid()
    while cur and cur > 1:
        if cur == pid:
            return True
        cur = _pid_ppid(cur)
    return False


def kill_plink(sys_=None, silent=True):
    global plink_pid, cmd_window_pid, linux_terminal_pid
    command = ''
    if plink_pid:
        if sys_ == 'Windows':
            command = f'taskkill /f /pid {plink_pid} /t'
            print(f"Terminating Plink with command: {command}")
            try:
                run_shell_cmd(command, silent=silent)
            except Exception as e:
                print(f"Error killing PID {plink_pid}: {e}")
            plink_pid = None
            _close_all_plink_windows_windows(silent=silent)
            return 0
        else:
            # Defend against PID reuse: only target plink_pid if it still names plink.
            comm = _pid_comm(plink_pid)
            if comm != 'plink':
                print(f"plink_pid {plink_pid} now names {comm!r}, not 'plink' — skipping targeted kill, falling back to pkill.")
                plink_pid = None
            else:
                ppid = _pid_ppid(plink_pid)

                # Killing the parent was meant to close the bash piping into plink, but if the
                # parent is us (or an ancestor — happens under PyCharm's subreaper), it would
                # kill the GUI. linux_terminal_pid below already handles the terminal window.
                if ppid and ppid > 1 and not _is_self_or_ancestor(ppid):
                    command = f'kill -9 {plink_pid} {ppid}'
                    print(f"Terminating Plink and parent terminal with command: {command}")
                else:
                    if ppid is not None and _is_self_or_ancestor(ppid):
                        print(f"Refusing to kill ppid {ppid}: it is this GUI or an ancestor.")
                    command = f'kill -9 {plink_pid}'
                    print(f"Terminating Plink with command: {command}")

                try:
                    run_shell_cmd(command, silent=silent)
                except Exception as e:
                    print(f"Error killing PID {plink_pid}: {e}")
                plink_pid = None

                # Kill the terminal window itself — qterminal and similar don't auto-close when bash dies
                if linux_terminal_pid:
                    try:
                        subprocess.run(['kill', '-9', str(linux_terminal_pid)], check=False)
                        print(f"Killed terminal PID: {linux_terminal_pid}")
                    except Exception as e:
                        print(f"Error killing terminal PID {linux_terminal_pid}: {e}")
                    linux_terminal_pid = None
                return 0

    # If we reached here, either plink_pid was None or we want to be sure
    command = ''
    if sys_ == 'Linux':
        command = 'pkill -e plink; pkill -f "plink-terminal-server"'
    elif sys_ == 'Windows':
        command = 'taskkill /f /im plink.exe'
    elif sys_ == 'Darwin':
        command = 'pkill plink'
    else:
        if sys_ is not None:
            print(f"kill_plink: SYS = {sys_} unknown")
        return -1

    print(f"Terminating Plink with command: {command}")
    if not silent:
        print(Colors.bg.brightblack, Colors.fg.wheat)
        result = run_shell_cmd(command, silent=silent)
        print(Colors.reset)
        if result == -1:
            print(Colors.fg.blue, 'failed.', Colors.reset)
            return None, False
    else:
        result = run_shell_cmd(command, silent=silent)

    if sys_ == 'Windows':
        _close_all_plink_windows_windows(silent=silent)

    return result


def look_plink(sys_=None, silent=True):
    if sys_ == 'Linux':
        try:
            output = subprocess.check_output(['pgrep', 'plink']).decode('ascii')
            return len(output.strip()) > 0
        except subprocess.CalledProcessError:
            return False
    elif sys_ == 'Windows':
        try:
            output = subprocess.check_output(['tasklist', '/FI', 'IMAGENAME eq plink.exe', '/NH']).decode('ascii')
            return 'plink.exe' in output.lower()
        except subprocess.CalledProcessError:
            return False
    elif sys_ == 'Darwin':
        try:
            output = subprocess.check_output(['pgrep', 'plink']).decode('ascii')
            return len(output.strip()) > 0
        except subprocess.CalledProcessError:
            return False
    else:
        print(f"look_plink: SYS = {sys_} unknown")
        return False


def lookup_macro():
    macro_name = macro_option.get()
    macro_data = macro_lookup.get(macro_name)
    if macro_data is None:
        print(f"Error: Macro '{macro_name}' not found in macro_lookup.")
        return

    dum_, macro_val, ev_val = macro_data
    macro.set(macro_val)
    macro_button.config(text=macro.get())
    while len(ev_val) < 4:
        ev_val = ev_val + ('',)
    if ev_val[0]:
        ev1_label.config(text='-' + ev_val[0])
    else:
        ev1_label.config(text='')
    if ev_val[1]:
        ev2_label.config(text='-' + ev_val[1])
    else:
        ev2_label.config(text='')
    if ev_val[2]:
        ev3_label.config(text='-' + ev_val[2])
    else:
        ev3_label.config(text='')
    if ev_val[3]:
        ev4_label.config(text='-' + ev_val[3])
    else:
        ev4_label.config(text='')


def lookup_start():
    option_name = option.get()
    option_data = lookup.get(option_name)
    if option_data is None:
        print(f"Error: Option '{option_name}' not found in lookup.")
        return

    dawdle_val_, start_val, ev_val = option_data
    start.set(start_val)
    start_button.config(text=start.get())
    while len(ev_val) < 4:
        ev_val = ev_val + ('',)
    if ev_val[0]:
        ev1_label.config(text='-' + ev_val[0])
    else:
        ev1_label.config(text='')
    if ev_val[1]:
        ev2_label.config(text='-' + ev_val[1])
    else:
        ev2_label.config(text='')
    if ev_val[2]:
        ev3_label.config(text='-' + ev_val[2])
    else:
        ev3_label.config(text='')
    if ev_val[3]:
        ev4_label.config(text='-' + ev_val[3])
    else:
        ev4_label.config(text='')
    timer_val.set(dawdle_val_)


def lookup_test():
    test_filename.set(plink_connection.get(Test.unit or '', ''))


def check_auto_plink():
    plink_path = Path(plink_test_csv_path.get())
    auto_plink_path = plink_path.parent / 'auto_plink.csv'
    if auto_plink_path.is_file():
        print(f"Acknowledged: {auto_plink_path} exists.")
    else:
        _write_default_auto_plink(auto_plink_path)
        print(f"Prepopulated {auto_plink_path} with default_auto content.")
    print(f"Report: plink_test.csv location is {plink_path}")


def _write_default_auto_plink(path):
    fields = [f.strip() for f in default_auto_header.split(',')]
    with open(path, 'w') as f:
        f.write('#' + default_auto_header + ',\n')
        for row in default_auto:
            f.write(', '.join(str(row.get(field, '')) for field in fields) + ',\n')


def plink_size():
    if Path(plink_test_csv_path.get()).is_file():
        enter_size = Path(plink_test_csv_path.get()).stat().st_size  # bytes
    else:
        enter_size = 0
    return enter_size


def close_auto_windows(close_figs=False):
    """Close plink terminal, PlotKiller and timer windows for AUTO mode.
    close_figs=True: close all figures (AUTO start, no handles yet).
    close_figs=False: close previous case's figures by handle, leaving the new case's figures intact."""
    global timer, auto_fig_list
    kill_plink(platform.system())
    _cancel_monitor_plink()
    if timer is not None:
        try:
            timer.close()
        except Exception:
            pass
        timer = None
    for widget in master.winfo_children():
        if isinstance(widget, tk.Toplevel):
            try:
                t = widget.title()
                if t in ('SOC-close', 'SOC-countdown'):
                    widget.destroy()
            except Exception:
                pass
    if close_figs:
        plt.close('all')
        auto_fig_list = None


def _get_putty_serial_line(session_name):
    """Return the SerialLine from the named putty session config, or None if not found."""
    if platform.system() in ('Linux', 'Darwin'):
        session_file = Path.home() / '.putty' / 'sessions' / session_name
        if session_file.is_file():
            for line in session_file.read_text(errors='replace').splitlines():
                if line.startswith('SerialLine='):
                    return line.split('=', 1)[1].strip()
    elif platform.system() == 'Windows':
        try:
            import winreg
            key_path = rf'Software\SimonTatham\PuTTY\Sessions\{session_name}'
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path) as key:
                value, _ = winreg.QueryValueEx(key, 'SerialLine')
                return value
        except Exception:
            pass
    return None


def grab_auto():
    global auto_running, auto_fig_list, auto_case_index, auto_case_total

    # Pre-flight checks before starting AUTO
    errors = []
    serial_device = _get_putty_serial_line(test_filename.get())
    if serial_device:
        if not Path(serial_device).exists():
            errors.append(f"USB serial device not connected: {serial_device}")
    else:
        errors.append(f"Could not read serial line from putty session '{test_filename.get()}'")
    dr_folder = Test.dataReduction_folder
    if not Path(dr_folder).exists():
        errors.append(f"dataReduction folder not found:\n  {dr_folder}")
    if errors:
        tkinter.messagebox.showerror(title="AUTO Prerequisites Not Met",
                                     message="\n\n".join(errors))
        return

    plink_path = Path(plink_test_csv_path.get())
    auto_plink_path = plink_path.parent / 'auto_plink.csv'
    if not auto_plink_path.is_file():
        print(f"Error: {auto_plink_path} not found.")
        tkinter.messagebox.showerror(title="File Not Found", message=f"Could not find {auto_plink_path}")
        return

    print(f"Reading {auto_plink_path}...")
    try:
        with open(auto_plink_path, 'r') as f:
            lines = f.readlines()
        
        header_line = None
        header_fields = []
        data_rows = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                if header_line is None:
                    header_line = line
                    header_fields = [f.strip() for f in line.lstrip('#').split(',')]
                    # Filter out empty fields that might occur if there's a trailing comma
                    header_fields = [f for f in header_fields if f]
                    print(f"Fields: {header_fields}")
                continue
            
            if header_line is not None:
                values = [v.strip() for v in line.split(',')][:len(header_fields)]
                if len(values) < len(header_fields):
                    print(f"Skipping line (too few fields {len(values)} vs {len(header_fields)}): {line}")
                    continue
                data_rows.append(values)

        if not data_rows:
            print("No valid data lines found in auto_plink.csv")
            return

        # Prepare display for the confirmation dialog
        display_lines = []
        for values in data_rows:
            display_parts = []
            for i in range(min(len(header_fields), len(values))):
                display_parts.append(f"{header_fields[i]}: {values[i]}")
            display_lines.append(" | ".join(display_parts))
        
        all_lines_text = "\n".join(display_lines)
        print(f"All configurations:\n{all_lines_text}")

        # Custom wide dialog
        dialog = tk.Toplevel(master)
        dialog.title("Automate Configurations?")
        dialog.geometry("1200x400")
        dialog.grab_set()
        dialog.transient(master)

        result = tk.BooleanVar(value=False)

        def on_yes():
            result.set(True)
            dialog.destroy()

        def on_no():
            result.set(False)
            dialog.destroy()

        # Prompt + Yes/No at the top so they stay visible even when the config list is tall.
        top_frame = tk.Frame(dialog)
        top_frame.pack(side='top', fill='x', padx=20, pady=10)
        prompt_label = tk.Label(top_frame, text="Do you want to run these configurations automatically?",
                                font=('Arial', 11, 'bold'))
        prompt_label.pack(side='left')
        btn_frame = tk.Frame(top_frame)
        btn_frame.pack(side='right')
        tk.Button(btn_frame, text="Yes", width=10, command=on_yes).pack(side='left', padx=10)
        tk.Button(btn_frame, text="No", width=10, command=on_no).pack(side='left', padx=10)

        msg_label = tk.Label(dialog, text=all_lines_text, justify='left', font=('Courier', 10))
        msg_label.pack(side='top', padx=20, pady=20, fill='both', expand=True)

        master.wait_window(dialog)
        
        if not result.get():
            print("User response for automatic run: No")
            return
        
        print("User response for automatic run: Yes")

        # Close all PlotKiller/timer windows and figures before starting AUTO
        close_auto_windows(close_figs=True)

        # Save configuration
        saved_config = {
            'folder': Test.dataReduction_folder,
            'version': Test.version,
            'unit': test_unit.get(),
            'battery': test_battery.get(),
            'macro': macro_option.get(),
            'option': option.get(),
        }
        print(f"Saved configuration: {saved_config}")

        def set_red(widget):
            try:
                widget.config(fg='red', activeforeground='red')
            except Exception:
                try:
                    widget.config(fg='red')
                except Exception:
                    pass

        auto_running = True

        # Process each line
        def process_next_config(index):
            global auto_running, auto_fig_list, auto_case_index, auto_case_total
            if index >= len(data_rows):
                n_cases = len(data_rows)
                auto_running = False

                # Restore runtime state
                Test.dataReduction_folder = saved_config['folder']
                Test.update_folder_button()
                Test.version = saved_config['version']
                Test.update_version_button()
                test_unit.set(saved_config['unit'])
                test_battery.set(saved_config['battery'])
                macro_option.set(saved_config['macro'])
                option.set(saved_config['option'])

                # Write folder and version back to .ini (traces handle the rest)
                Test.cf[Test.ind]['dataReduction_folder'] = saved_config['folder']
                Test.cf[Test.ind]['version'] = saved_config['version']
                Test.cf.save_to_file()

                # Restore colors to pre-AUTO state
                Test.folder_button.config(fg='blue', activeforeground='blue')
                Test.version_button.config(fg='blue', activeforeground='blue')
                Test.unit_button.config(fg='black', activeforeground='black')
                Test.battery_button.config(fg='black', activeforeground='black')
                macro_sel.config(fg='black', activeforeground='black')
                sel.config(fg='black', activeforeground='black')
                sel1.config(fg='black', activeforeground='black')

                lookup_start()
                print(f"AUTO complete: {n_cases} case(s) run. Original configuration restored.")
                tkinter.messagebox.showinfo("AUTO Complete",
                                            f"{n_cases} case(s) run.\nOriginal configuration restored.")
                if auto_fig_list:
                    string = 'plots ' + str(auto_fig_list[0].number) + ' - ' + str(auto_fig_list[-1].number)
                    show_killer(string, 'AUTO', fig_list=auto_fig_list)
                return

            values = data_rows[index]
            # Map values to fields
            config = {}
            for i in range(min(len(header_fields), len(values))):
                config[header_fields[i]] = values[i]

            print(f"Processing configuration {index+1}/{len(data_rows)}: {config}")

            if 'folder' in config:
                Test.dataReduction_folder = config['folder']
                Test.update_folder_button()
                set_red(Test.folder_button)

            if 'version' in config:
                Test.version = config['version']
                Test.update_version_button()
                set_red(Test.version_button)

            if 'unit' in config:
                test_unit.set(config['unit'])
                set_red(Test.unit_button)

            if 'battery' in config:
                test_battery.set(config['battery'])
                set_red(Test.battery_button)

            if 'macro' in config:
                if config['macro'] in macro_lookup:
                    macro_option.set(config['macro'])
                    set_red(macro_sel)
                elif config['macro'] in lookup:
                    # If it's in 'lookup' but not 'macro_lookup', it's likely intended as an 'option'
                    option.set(config['macro'])
                    set_red(sel)
                    set_red(sel1)
                    lookup_start()
                else:
                    print(f"Error: Macro '{config['macro']}' not found in macro_lookup or lookup. Skipping.")

            # Track AUTO case progress (used by start_plink for status print)
            auto_case_index = index
            auto_case_total = len(data_rows)

            # Trigger START HERE button
            print(f"\n\nTriggering START HERE for config {index+1}")
            grab_init(force_if_ready=True, force_kill=True)

            # Wait for READY, then click START BUTTON, then wait for DONE
            def check_ready_and_start():
                if Path(plink_test_csv_path.get()).is_file():
                    try:
                        with open(plink_test_csv_path.get(), 'rb') as f:
                            f.seek(0, 2)
                            size = f.tell()
                            f.seek(max(0, size - 1024))
                            last_data = f.read().decode('utf-8', errors='ignore')
                            if '***READY***' in last_data:
                                print(f"***READY*** detected for config {index+1}. Triggering start_button.")
                                grab_start()
                                # After starting, wait for DONE
                                master.after(1000, check_completion)
                                return
                    except Exception as e:
                        print(f"Error monitoring plink file for READY in AUTO: {e}")
                
                master.after(1000, check_ready_and_start)

            def check_completion():
                if Path(plink_test_csv_path.get()).is_file():
                    try:
                        with open(plink_test_csv_path.get(), 'rb') as f:
                            f.seek(0, 2)
                            size = f.tell()
                            f.seek(max(0, size - 1024))
                            last_data = f.read().decode('utf-8', errors='ignore')
                            if '***DONE***' in last_data:
                                print(f"***DONE*** detected for config {index+1}")
                                close_auto_windows(close_figs=False)
                                save_data(show_killer_=False)
                                master.after(1000, lambda: process_next_config(index + 1))
                                return
                    except Exception as e:
                        print(f"Error monitoring plink file for DONE in AUTO: {e}")
                
                master.after(1000, check_completion)

            master.after(1000, check_ready_and_start)

        # Start the sequential processing
        process_next_config(0)

    except Exception as e:
        print(f"Error reading auto_plink.csv: {e}")
        tkinter.messagebox.showerror(title="Read Error", message=f"Error reading auto_plink.csv: {e}")


def ref_remove():
    top_panel_right.pack_forget()
    run_x_button.config(text='Compare Run Sim')
    run_sim_hist_button.config(text='Run Both of These')
    hist_sim_button.config(text='Compare Hist Sim')
    hist_hist_button.forget()
    hist_sim_button.pack(side='left', padx=5, pady=5)
    run_sim_hist_button.pack(side='right', padx=5, pady=5)
    Ref.label.forget()


def ref_restore():
    top_panel_right.pack(expand=True, fill='both')
    run_x_button.config(text='Compare Run Run')
    run_sim_hist_button.forget()
    hist_sim_button.forget()
    hist_hist_button.pack(side='left', padx=5, pady=5)
    Ref.label.pack(padx=5, pady=5)


def save_data(show_killer_=True):
    global timer, auto_fig_list
    _was_auto = auto_running  # snapshot before any async callback can flip it
    print(f"save_data: {plink_test_csv_path.get()=}  auto_running={_was_auto}")
    if size_of(plink_test_csv_path.get()) > 64:  # bytes
        # For custom option, redefine Test.file_path if requested
        new_file_txt = None
        if option.get() == 'custom':
            new_file_txt = tk.simpledialog.askstring(title=__file__, prompt="custom file name string:")
            if new_file_txt is not None:
                Test.create_file_path_and_key(name_override=new_file_txt)
                Test.label.config(text=Test.file_txt)
                print('Test.file_path', Test.file_path)
        if Path(Test.file_path).is_file() and Path(Test.file_path).stat().st_size > 0:  # bytes
            if auto_overwrite.get() or _was_auto:
                print('auto over-write enabled')
                p = Path(Test.file_path)
                ts = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
                renamed = p.with_name(p.stem + '_' + ts + p.suffix)
                p.rename(renamed)
                print(f"Renamed existing gdrive file to {renamed.name}")
            else:
                confirmation = tk.messagebox.askyesno('query overwrite', 'File exists:  overwrite?')
                if not confirmation:
                    print('skipped overwrite')
                    tkinter.messagebox.showwarning(message='retained ' + Test.file_path)
                    return
        save_data_button.config(bg='yellow', activebackground='yellow', fg='black', activeforeground='black',
                                text='data saving')
        tksleep(0.1)
        if run_start_time is not None:
            elapsed = time.time() - run_start_time
            print(f"Run elapsed: {elapsed:.1f}s")
            with open(plink_test_csv_path.get(), 'a') as _ef:
                _ef.write(f'#elapsed_s,{elapsed:.1f}\n')
        copy_clean(plink_test_csv_path.get(), Test.file_path)
        print('copied ', plink_test_csv_path.get(), '\nto\n', Test.file_path)
        if timer is not None:
            timer.close()
            timer = None
        save_data_button.config(bg='green', activebackground='green', fg='red', activeforeground='red',
                                text='data saved')
        empty_file(plink_test_csv_path.get())
        print('updating Test file label')
        Test.create_file_path_and_key(name_override=new_file_txt)
        Test.update_key_label()  # refresh key_exists_in_file from the newly saved file
        _was_auto = auto_running  # capture before process_next_config may clear it
        if auto_overwrite.get() or _was_auto:
            print('auto over-write triggering comparison')
            # In AUTO mode close the previous case's figures now, before new ones are created
            if auto_running and auto_fig_list is not None:
                for _fig in auto_fig_list:
                    try:
                        plt.close(_fig)
                    except Exception:
                        pass
                auto_fig_list = None
            result = compare_run(show_killer_=show_killer_)
            if result is not None:
                auto_fig_list = result[0]
    else:
        print('plink test file non-existent or too small (<64 bytes) probably already done')
        tkinter.messagebox.showwarning(message="Nothing to save")
    start_button.config(bg='black', activebackground='black', fg='#00ff00', activeforeground='#00ff00')


def save_data_as():
    global timer
    if size_of(plink_test_csv_path.get()) > 512:  # bytes
        # For custom option, redefine Test.file_path if requested
        if option.get() == 'custom':
            new_file_txt = tk.simpledialog.askstring(title=__file__, prompt="custom file name string:")
            if new_file_txt is not None:
                Test.create_file_path_and_key(name_override=new_file_txt)
                Test.label.config(text=Test.file_txt)
                print('Test.file_path', Test.file_path)
        else:
            new_file_txt = tk.simpledialog.askstring(title=__file__, prompt="custom file name string:",
                                                     initialvalue=Test.file_txt)
            if new_file_txt is not None:
                Test.create_file_path_and_key(name_override=new_file_txt)
                Test.label.config(text=Test.file_txt)
                print('Test.file_path', Test.file_path)
        if Path(Test.file_path).is_file() and Path(Test.file_path).stat().st_size > 0:  # bytes
            confirmation = tk.messagebox.askyesno('query overwrite', 'File exists:  overwrite?')
            if not confirmation:
                print('reset and use clear')
                tkinter.messagebox.showwarning(message='reset and use clear')
                return
        save_data_as_button.config(bg='yellow', activebackground='yellow', fg='black', activeforeground='black',
                                   text='data saving')
        tksleep(0.1)
        copy_clean(plink_test_csv_path.get(), Test.file_path)
        print('copied ', plink_test_csv_path.get(), '\nto\n', Test.file_path)
        if timer is not None:
            timer.close()
            timer = None
        save_data_as_button.config(bg='green', activebackground='green', fg='red', activeforeground='red',
                                   text='data saved as')
        empty_file(plink_test_csv_path.get())
        print('updating Test file label')
        Test.create_file_path_and_key(name_override=new_file_txt)
    else:
        print('plink test file is too small (<512 bytes) probably already done')
        tkinter.messagebox.showwarning(message="Nothing to save")
    start_button.config(bg='black', activebackground='black', fg='#00ff00', activeforeground='#00ff00')


def save_progress():
    global timer
    if size_of(plink_test_csv_path.get()) > 64:  # bytes
        # For custom option, redefine Test.file_path if requested
        new_file_txt = None
        if option.get() == 'custom':
            new_file_txt = tk.simpledialog.askstring(title=__file__, prompt="custom file name string:")
            if new_file_txt is not None:
                Test.create_file_path_and_key(name_override=new_file_txt)
                Test.label.config(text=Test.file_txt)
                print('Test.file_path', Test.file_path)
        if Path(Test.file_path).is_file() and Path(Test.file_path).stat().st_size > 0:  # bytes
            confirmation = tk.messagebox.askyesno('query overwrite', 'File exists:  overwrite?')
            if not confirmation:
                print('skipped overwrite')
                tkinter.messagebox.showwarning(message='Nothing changed')
                return
        save_progress_button.config(bg='yellow', activebackground='yellow', fg='black', activeforeground='black',
                                    text='data saving')
        tksleep(0.1)
        copy_clean(plink_test_csv_path.get(), Test.file_path)
        if timer is not None:
            timer.close()
            timer = None
        save_progress_button.config(bg=bg_color, activebackground=bg_color, fg='black', activeforeground='black',
                                    text='save_progress')
        print('copied ', plink_test_csv_path.get(), '\nto\n', Test.file_path)
        print('updating Test file label')
        Test.create_file_path_and_key(name_override=new_file_txt)
        tkinter.messagebox.showwarning(message="Progress saved")
        update_data_buttons()
    else:
        print('plink test file non-existent or too small (<64 bytes) probably already done')
        tkinter.messagebox.showwarning(message="Nothing to save")


def save_plink():
    m_str = datetime.datetime.fromtimestamp(Path(plink_test_csv_path.get()).stat().st_mtime).strftime("%Y-%m-%dT%H-%M-%S").replace(' ', 'T')
    plink_test_sav_path = tk.StringVar(master, str(PurePosixPath(path_to_temp.get()) / ('plink_' + m_str + '.csv')))
    print(f"GUI_PlinkSOC(save_plink):\n{plink_test_csv_path.get()=}\n{plink_test_sav_path.get()=}\n")
    try:
        shutil.copyfile(plink_test_csv_path.get(), plink_test_sav_path.get())
        print('wrote', plink_test_sav_path.get())
        empty_file(plink_test_csv_path.get())
        return True
    except PermissionError:
        print('plink holding file open')
        return False


def is_plink_ready():
    if Path(plink_test_csv_path.get()).is_file():
        try:
            with open(plink_test_csv_path.get(), 'rb') as f:
                f.seek(0, 2)
                size = f.tell()
                # Read last 1024 bytes to check for ***READY***
                f.seek(max(0, size - 1024))
                last_data = f.read().decode('utf-8', errors='ignore')
                # Check for ***READY*** with an extra line feed as requested
                if '***READY***' in last_data:
                    return True
        except Exception as e:
            print(f"Error checking plink ready: {e}")
    return False


def start_plink(command_to_paste=None, force_if_ready=False, force_kill=False, fg_color='#00ff00', bg_color='#000000'):
    global plink_pid, cmd_window_pid, cmd_window_pids, linux_terminal_pid
    lookup_test()
    if look_plink(platform.system()):
        if force_kill:
            print("AUTO: force-killing plink regardless of state.")
            kill_plink(platform.system())
            tksleep(0.5)
            empty_file(plink_test_csv_path.get())
            # Proceed to restart logic below
        elif force_if_ready:
            if is_plink_ready():
                print("Plink is READY. Restarting to automate command.")
                kill_plink(platform.system())
                tksleep(0.5)  # Give some time for the process to exit
                empty_file(plink_test_csv_path.get())
                # Proceed to restart logic below
            else:
                print("Plink is already open but NOT READY.")
                tkinter.messagebox.showinfo(title="Not Ready",
                                           message="Please wait until terminal is READY or run the START HERE button.")
                return False
        else:
            print("Plink already open. Killing and restarting as commanded.")
            kill_plink(platform.system())
            tksleep(0.5)
            empty_file(plink_test_csv_path.get())
            # Proceed to restart logic below

    enter_size = plink_size()
    if enter_size >= 64:
        if not save_plink():
            tkinter.messagebox.showwarning(message="plink may be open already")
        enter_size = plink_size()

    if enter_size < 64:
        kill_plink(platform.system())
        print(f'restarting plink   plink -load {test_filename.get()}')
        if platform.system() == 'Linux':
            term = shutil.which('gnome-terminal') or shutil.which('xterm') or 'x-terminal-emulator'
            # Use bash -c for an interactive window that pipes output to tee and stays open
            # User provided: gnome-terminal -- bash -c 'plink -load testsoc3p2 | tee ~/.local/plink_test.csv; exec bash'
            plink_base_cmd = f"plink -batch -T -load {test_filename.get()} | tee {plink_test_csv_path.get()}"
            if command_to_paste:
                # (echo 'command'; cat) | plink ... ensures the command is sent and the session remains interactive
                plink_cmd = f"(echo '{command_to_paste}'; cat) | {plink_base_cmd}; exec bash"
            else:
                plink_cmd = f"{plink_base_cmd}; exec bash"

            if 'gnome-terminal' in term:
                # zoom 0.8 is roughly "two sizes smaller" (standard is 1.0, 0.9 is one size, 0.8 is two)
                cmd = [term, '--app-id=local.plink-terminal-server', '--zoom=0.8', '--', 'bash', '-c',
                       f"echo -e '\\e]11;{bg_color}\\a\\e]10;{fg_color}\\a\\e]0;plink-terminal-server\\a'; clear; {plink_cmd}"]
                print(f"Running command: {shlex.join(cmd)}")
                proc = subprocess.Popen(cmd)
                linux_terminal_pid = proc.pid
                tksleep(1.0) # Wait for terminal to spawn plink
                try:
                    # Debug: Print full result of pgrep and ps -ef | grep plink
                    pgrep_search = f"plink -batch -T -load {test_filename.get()}"
                    print(f"Debug: pgrep -a -f result for '{pgrep_search}':")
                    try:
                        pgrep_out = subprocess.check_output(['pgrep', '-a', '-f', pgrep_search]).decode()
                        print(pgrep_out)
                    except subprocess.CalledProcessError:
                        print("No processes found with pgrep.")

                    # Find the newest plink process matching our session
                    out = subprocess.check_output(['pgrep', '-n', '-f', pgrep_search]).decode().strip()
                    if out:
                        plink_pid = int(out)
                except Exception:
                    pass  # plink_pid stays None; kill_plink falls back to pkill
                
                # Get the parent PID using ps
                ppid = "Unknown"
                try:
                    ppid_out = subprocess.check_output(['ps', '-o', 'ppid=', '-p', str(plink_pid)]).decode().strip()
                    if ppid_out:
                        ppid = ppid_out
                except Exception:
                    pass
                print(f"Spawned PID: {plink_pid}  PPID: {ppid}")
                if auto_running:
                    print(f"AUTO running case No. {auto_case_index + 1} of {auto_case_total}")
            elif 'xterm' in term:
                # xterm -bg black -fg green -fs 10 (assuming default is ~12)
                # Pass bash -c args separately so single quotes inside plink_cmd don't break the shell
                cmd = [term, '-T', 'plink-terminal-server', '-bg', bg_color, '-fg', fg_color, '-fs', '10', '-e', 'bash', '-c', plink_cmd]
                print(f"Running command: {shlex.join(cmd)}")
                proc = subprocess.Popen(cmd)
                linux_terminal_pid = proc.pid
                tksleep(1.0) # Wait for terminal to spawn plink
                try:
                    # Debug: Print full result of pgrep and ps -ef | grep plink
                    pgrep_search = f"plink -batch -T -load {test_filename.get()}"
                    print(f"Debug: pgrep -a -f result for '{pgrep_search}':")
                    try:
                        pgrep_out = subprocess.check_output(['pgrep', '-a', '-f', pgrep_search]).decode()
                        print(pgrep_out)
                    except subprocess.CalledProcessError:
                        print("No processes found with pgrep.")

                    out = subprocess.check_output(['pgrep', '-n', '-f', pgrep_search]).decode().strip()
                    if out:
                        plink_pid = int(out)
                except Exception:
                    pass  # plink_pid stays None; kill_plink falls back to pkill
                
                # Get the parent PID using ps
                ppid = "Unknown"
                try:
                    ppid_out = subprocess.check_output(['ps', '-o', 'ppid=', '-p', str(plink_pid)]).decode().strip()
                    if ppid_out:
                        ppid = ppid_out
                except Exception:
                    pass
                print(f"Spawned PID: {plink_pid}  PPID: {ppid}")
                if auto_running:
                    print(f"AUTO running case No. {auto_case_index + 1} of {auto_case_total}")
            else:
                # qterminal / x-terminal-emulator: pass bash -c args separately to avoid single-quote
                # conflicts when plink_cmd contains quoted strings, and use OSC sequences for colors
                full_bash_cmd = f"echo -e '\\e]11;{bg_color}\\a\\e]10;{fg_color}\\a\\e]0;plink-terminal-server\\a'; clear; {plink_cmd}"
                cmd = [term, '-e', 'bash', '-c', full_bash_cmd]
                print(f"Running command: {shlex.join(cmd)}")
                proc = subprocess.Popen(cmd)
                linux_terminal_pid = proc.pid
                tksleep(1.0) # Wait for terminal to spawn plink
                try:
                    # Debug: Print full result of pgrep and ps -ef | grep plink
                    pgrep_search = f"plink -batch -T -load {test_filename.get()}"
                    print(f"Debug: pgrep -a -f result for '{pgrep_search}':")
                    try:
                        pgrep_out = subprocess.check_output(['pgrep', '-a', '-f', pgrep_search]).decode()
                        print(pgrep_out)
                    except subprocess.CalledProcessError:
                        print("No processes found with pgrep.")

                    out = subprocess.check_output(['pgrep', '-n', '-f', pgrep_search]).decode().strip()
                    if out:
                        plink_pid = int(out)
                except Exception:
                    pass  # plink_pid stays None; kill_plink falls back to pkill
                
                # Get the parent PID using ps
                ppid = "Unknown"
                try:
                    ppid_out = subprocess.check_output(['ps', '-o', 'ppid=', '-p', str(plink_pid)]).decode().strip()
                    if ppid_out:
                        ppid = ppid_out
                except Exception:
                    pass
                print(f"Spawned PID: {plink_pid}  PPID: {ppid}")
                if auto_running:
                    print(f"AUTO running case No. {auto_case_index + 1} of {auto_case_total}")
        elif platform.system() == 'Windows':
            # plink has no -tee flag.  Git's tee.exe opens with deny-read sharing so check_completion
            # can never read plink_test.csv while tee is running.  Use a tiny Python helper instead:
            # os.open() on Windows defaults to _SH_DENYNO, so other processes can read freely.
            csv_path = plink_test_csv_path.get()
            if fg_color == '#00ff00':
                win_color = '0A'  # black background, bright green (start_button)
            elif fg_color == '#ffffff':
                win_color = '0F'  # black background, white (init_button)
            else:
                win_color = '8E'  # dark gray background, light yellow wheat (AUTO)
            _tee_py = os.path.join(os.environ.get('TEMP', os.environ.get('TMP', 'C:\\Temp')), 'plink_tee.py')
            with open(_tee_py, 'w') as _tf:
                _tf.write(
                    'import sys, os\n'
                    'fd = os.open(sys.argv[1], os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_BINARY)\n'
                    'try:\n'
                    '    while True:\n'
                    '        chunk = os.read(0, 4096)\n'
                    '        if not chunk:\n'
                    '            break\n'
                    '        os.write(fd, chunk)\n'
                    '        os.write(1, chunk)\n'
                    'finally:\n'
                    '    os.close(fd)\n'
                )
            py_exe = sys.executable
            plink_base = f'plink -batch -T -load {test_filename.get()} | "{py_exe}" "{_tee_py}" "{csv_path}"'
            if command_to_paste:
                # Write command to a temp file so cmd.exe never sees the special chars (<>|&^)
                _cmd_file = os.path.join(os.environ.get('TEMP', os.environ.get('TMP', 'C:\\Temp')), 'plink_cmd.txt')
                with open(_cmd_file, 'w', newline='') as _cf:
                    _cf.write(command_to_paste + '\r\n')
                plink_cmd_bat = f'(type "{_cmd_file}" & type CON) | {plink_base}'
            else:
                plink_cmd_bat = plink_base
            # Unique window title so we can fall back to taskkill /fi "WINDOWTITLE eq ..." if PID kill fails
            _win_title = f'{cmd_window_title_prefix}_{os.getpid()}_{int(time.time()*1000)}'
            # Write to a temp batch file — avoids all cmd.exe quoting issues with paths containing spaces
            # Use a unique bat name per launch so concurrent windows don't collide on the file
            bat_path = os.path.join(os.environ.get('TEMP', os.environ.get('TMP', 'C:\\Temp')),
                                    f'plink_run_{int(time.time()*1000)}.bat')
            with open(bat_path, 'w') as _bf:
                _bf.write(f'@title {_win_title}\r\n@color {win_color}\r\n{plink_cmd_bat}\r\n')
            # CREATE_NEW_CONSOLE opens a real separate window and proc.pid IS that window's PID
            print(f"Running command: cmd /k {bat_path}  (title={_win_title})")
            proc = subprocess.Popen(['cmd', '/k', bat_path],
                                    creationflags=subprocess.CREATE_NEW_CONSOLE)
            cmd_window_pid = proc.pid
            cmd_window_pids.add(proc.pid)
            tksleep(1.0)
            try:
                # Find the newest plink process
                out = subprocess.check_output(['tasklist', '/FI', 'IMAGENAME eq plink.exe', '/NH', '/FO', 'CSV']).decode('ascii')
                # tasklist output in CSV: "plink.exe","1234","Console","1","5,678 K"
                lines = out.strip().split('\n')
                if lines:
                    last_line = lines[-1]
                    parts = last_line.split(',')
                    if len(parts) > 1:
                        plink_pid = int(parts[1].strip('"'))
            except Exception:
                pass  # plink_pid stays None; kill_plink falls back to taskkill /im plink.exe
            print(f"Spawned plink PID: {plink_pid}  cmd window PID: {cmd_window_pid}")
            if auto_running:
                print(f"AUTO running case No. {auto_case_index + 1} of {auto_case_total}")
        elif platform.system() == 'Darwin':
             # plink has no -tee flag; pipe stdout through shell tee instead
             if command_to_paste:
                 plink_cmd = f"(echo '{command_to_paste}'; cat) | plink -batch -T -load {test_filename.get()} | tee {plink_test_csv_path.get()}"
             else:
                 plink_cmd = f"plink -batch -T -load {test_filename.get()} | tee {plink_test_csv_path.get()}"

             script = (f'tell application "Terminal" to do script '
                       f'"printf \\"\\\\e]11;#000000\\\\a\\\\e]10;#00ff00\\\\a\\\\e]0;plink-terminal-server\\\\a\\"; clear; '
                       f'{plink_cmd}"\n'
                       f'tell application "Terminal" to set font size of window 1 to 10')
             cmd = ['osascript', '-e', script]
             print(f"Running command: {shlex.join(cmd)}")
             proc = subprocess.Popen(cmd)
             tksleep(1.0)
             try:
                 out = subprocess.check_output(['pgrep', '-n', '-f', f"plink -batch -T -load {test_filename.get()}"]).decode().strip()
                 if out:
                     plink_pid = int(out)
             except Exception:
                 pass  # plink_pid stays None; kill_plink falls back to pkill
             
             # Get the parent PID using ps
             ppid = "Unknown"
             try:
                 ppid_out = subprocess.check_output(['ps', '-o', 'ppid=', '-p', str(plink_pid)]).decode().strip()
                 if ppid_out:
                     ppid = ppid_out
             except Exception:
                 pass
             print(f"Spawned PID: {plink_pid}  PPID: {ppid}")
             if auto_running:
                 print(f"AUTO running case No. {auto_case_index + 1} of {auto_case_total}")
    return True


def start_timer():
    global timer
    timer = CountdownTimer(master, timer_val.get(), max_flash=60, exit_function=None, trigger=True)


def swap_run_test():
    """Swap and save Test and Ref choices"""
    global Test, Ref
    swap = Test.__copy__()
    Test.super_shallow_copy(Ref)
    Ref.super_shallow_copy(swap)
    test_unit.set(Test.unit)  # does Test update
    ref_unit.set(Ref.unit)  # does Ref update
    test_battery.set(Test.battery)  # does Test update
    ref_battery.set(Ref.battery)  # does Ref update


def tksleep(t):
    """emulating time.sleep(seconds)"""
    ms = int(t*1000)
    var = tk.IntVar(master)
    var.set(0)
    master.after(ms, var.set, 1)
    master.wait_variable(var)


def update_data_buttons():
    save_data_button.config(bg=bg_color, activebackground=bg_color, fg='black', activeforeground='black',
                            text='save data')
    save_data_as_button.config(bg=bg_color, activebackground=bg_color, fg='black', activeforeground='black',
                               text='save data as')
    start_button.config(bg='black', activebackground='black', fg='#00ff00', activeforeground='#00ff00')


if __name__ == '__main__':  # Example usage.  Ran ok 20260217
    import os
    import tkinter as tk
    from tkinter import ttk

    ex_root = ExRoot()

    cf = Begini(__file__, default_dict)

    # Define frames
    min_width = 800
    main_height = 500
    folder_reveal = 25
    wrap_length = 500
    wrap_length_note = 700
    note_font = ("Arial bold", 10)
    label_font = ("Arial bold", 12)
    label_font_gentle = ("Arial", 10)
    butt_font = ("Arial", 8)
    butt_font_large = ("Arial bold", 10)
    bg_color = "lightgray"

    # Master and header
    print("creating master")
    master = tk.Tk(className='GUI_PlinkSOC')
    print("master created")
    master.title('State of Charge (Plink)')
    master.wm_minsize(width=min_width, height=main_height)
    timer = None
    print("creating Ref")
    Ref = Exec(cf, 'ref', path_disp_len_=folder_reveal)
    print("Ref created")
    print("creating Test")
    Test = Exec(cf, 'test', path_disp_len_=folder_reveal)
    print("Test created")
    if platform.system() == 'Linux':
        plink_test_csv_path = tk.StringVar(master, '/home/daveg/.local/plink_test.csv')
        path_to_temp = tk.StringVar(master, '/home/daveg/.local')
    elif platform.system() == 'Darwin':
        plink_test_csv_path = tk.StringVar(master, '/Users/daveg/.local/plink_test.csv')
        path_to_temp = tk.StringVar(master, '/Users/daveg/.local')
    else:
        local_app_data_ = os.getenv('LOCALAPPDATA') or str(Path.home() / 'AppData' / 'Local')
        plink_test_csv_path = tk.StringVar(master, str(Path(local_app_data_) / 'Temp' / 'plink_test.csv'))
        path_to_temp = tk.StringVar(master, str(Path(local_app_data_) / 'Temp'))
    print(f"{plink_test_csv_path.get()=}")
    check_auto_plink()
    print("loading icon")
    icon_path = str(PurePosixPath(ex_root.script_loc) / 'GUI_TestSOC.png')
    _icon_photo = tk.PhotoImage(file=icon_path)
    master.iconphoto(False, _icon_photo)
    print("icon loaded")
    top_panel = tk.Frame(master)
    top_panel.pack(expand=True, fill='both')
    top_panel_left = tk.Frame(top_panel)
    top_panel_left.pack(side='left', expand=True, fill='both')
    top_panel_left_ctr = tk.Frame(top_panel)
    top_panel_left_ctr.pack(side='left', expand=True, fill='both')
    top_panel_right_ctr = tk.Frame(top_panel)
    top_panel_right_ctr.pack(side='left', expand=True, fill='both')
    top_panel_right = tk.Frame(top_panel)
    top_panel_right.pack(side='left', expand=True, fill='both')

    # Test/modeling row
    tk.Label(top_panel_left, text="Item", fg="blue", font=label_font).pack(pady=2)
    tk.Label(top_panel_left_ctr, text="Test", fg="blue", font=label_font).pack(pady=2)
    model_str = cf['others']['modeling']
    if model_str == 'True':
        modeling = tk.BooleanVar(master, True)
    else:
        modeling = tk.BooleanVar(master, False)
    modeling_button = tk.Checkbutton(top_panel_right_ctr, text='modeling', bg=bg_color, variable=modeling,
                                     onvalue=True, offvalue=False)
    modeling_button.pack(pady=2, fill='x')
    modeling.trace_add('write', handle_modeling)
    ref_label = tk.Label(top_panel_right, text="Ref", fg="blue", font=label_font)
    ref_label.pack(pady=2, expand=True, fill='both')

    # Folder row
    working_label = tk.Label(top_panel_left, text="dataReduction Folder", font=label_font)
    Test.folder_button = myButton(top_panel_left_ctr, text=Test.dataReduction_folder[-folder_reveal:],
                                  command=Test.enter_data_reduction_folder,
                                  fg="blue", bg=bg_color)
    auto_overwrite_str = cf['others'].get('auto_overwrite', 'False')
    if auto_overwrite_str == 'True':
        auto_overwrite = tk.BooleanVar(master, True)
    else:
        auto_overwrite = tk.BooleanVar(master, False)
    auto_overwrite_button = tk.Checkbutton(top_panel_right_ctr, text='auto over-write', bg=bg_color,
                                           variable=auto_overwrite, onvalue=True, offvalue=False)
    auto_overwrite_button.pack(pady=2, fill='x')
    auto_overwrite.trace_add('write', handle_auto_overwrite)

    Ref.folder_button = myButton(top_panel_right, text=Ref.dataReduction_folder[-folder_reveal:],
                                 command=Ref.enter_data_reduction_folder,
                                 fg="blue", bg=bg_color)
    working_label.pack(padx=5, pady=5)
    Test.folder_button.pack(padx=5, pady=5, anchor='w')
    Ref.folder_button.pack(padx=5, pady=5, anchor='e')

    # Version row
    tk.Label(top_panel_left, text="Version", font=label_font).pack(pady=2)
    Test.version_button = myButton(top_panel_left_ctr, text=Test.version, command=Test.enter_version, fg="blue", bg=bg_color)
    Test.version_button.pack(pady=2)
    Ref.version_button = myButton(top_panel_right, text=Ref.version, command=Ref.enter_version, fg="blue", bg=bg_color)
    Ref.version_button.pack(pady=2)

    # Unit row
    tk.Label(top_panel_left, text="Unit", font=label_font).pack(pady=2, expand=True, fill='both')
    test_unit = tk.StringVar(master, Test.unit)
    Test.unit_button = tk.OptionMenu(top_panel_left_ctr, test_unit, *unit_list)
    test_unit.trace_add('write', handle_test_unit)
    Test.unit_button.pack(pady=2)
    ref_unit = tk.StringVar(master, Ref.unit)
    Ref.unit_button = tk.OptionMenu(top_panel_right, ref_unit, *unit_list)
    ref_unit.trace_add('write', handle_run_unit)
    Ref.unit_button.pack(pady=2)
    
    test_filename = tk.StringVar(master, plink_connection.get(Test.unit or '', ''))

    # Battery row
    tk.Label(top_panel_left, text="Battery", font=label_font).pack(pady=2, expand=True, fill='both')
    test_battery = tk.StringVar(master, Test.battery)
    Test.battery_button = tk.OptionMenu(top_panel_left_ctr, test_battery, *battery_list)
    test_battery.trace_add('write', handle_test_battery)
    Test.battery_button.pack(pady=2)
    ref_battery = tk.StringVar(master, Ref.battery)
    Ref.battery_button = tk.OptionMenu(top_panel_right, ref_battery, *battery_list)
    ref_battery.trace_add('write', handle_run_battery)
    Ref.battery_button.pack(pady=2)

    # Key row
    tk.Label(top_panel_left, text="Key", font=label_font).pack(pady=2, expand=True, fill='both')
    Test.key_label = tk.Label(top_panel_left_ctr, text=Test.key)
    Test.key_label.pack(padx=5, pady=5)
    Ref.key_label = tk.Label(top_panel_right, text=Ref.key)
    Ref.key_label.pack(padx=5, pady=5)

    # Swap row
    tk.Label(top_panel_left, text="", font=label_font).pack(pady=2, expand=True, fill='both')
    tk.Label(top_panel_left_ctr, text="", font=label_font).pack(pady=2, expand=True, fill='both')
    swap_button = myButton(top_panel_right, text="swap Ref<-->Test", command=swap_run_test, bg=bg_color)
    swap_button.pack(side='right', padx=5, pady=5)

    # Image — reuse the PhotoImage already loaded for the window icon
    picture = _icon_photo.subsample(5, 5)
    label = tk.Label(top_panel_right_ctr, image=picture)
    label.pack(padx=5, pady=5, expand=True, fill='both')

    # Option panel
    option_sep_panel = tk.Frame(master)
    option_sep_panel.pack(expand=True, fill='x')
    tk.Label(option_sep_panel, text=' ', font=("Courier", 2), bg='darkgray').pack(expand=True, fill='x')
    option_panel = tk.Frame(master)
    option_panel.pack(expand=True, fill='both')
    option_panel_left = tk.Frame(option_panel)
    option_panel_left.pack(side='left', fill='x')
    option_panel_ctr = tk.Frame(option_panel)
    option_panel_ctr.pack(side='left', expand=True, fill='both')
    option_panel_right = tk.Frame(option_panel)
    option_panel_right.pack(side='left', expand=True, fill='both')

    # Option row
    option = tk.StringVar(master, str(cf['others']['option']))
    option_show = tk.StringVar(master, str(cf['others']['option']))
    sel = tk.OptionMenu(option_panel_left, option, *sel_list)
    sel.config(width=20, font=butt_font)
    sel.pack(padx=5, pady=5)
    sel1 = tk.OptionMenu(option_panel_left, option, *sel_list1)
    sel1.config(width=20, font=butt_font)
    sel1.pack(padx=5, pady=5)
    option.trace_add('write', handle_option)
    Test.label = tk.Label(option_panel_ctr, text=Test.file_txt)
    Test.label.pack(padx=5, pady=5, anchor='w')
    Ref.label = tk.Label(option_panel_right, text=Ref.file_txt)
    Ref.label.pack(padx=5, pady=5, anchor='e')
    Test.create_file_path_and_key(cf['others']['option'])
    Ref.create_file_path_and_key(cf['others']['option'])

    _, init_val, _ = lookup.get('satInit')
    init_row_frame = tk.Frame(option_panel_ctr)
    if platform.system() == 'Darwin':
        init_button = myButton(init_row_frame, text='START HERE and PASTE then\n wait for temp init complete', command=grab_init, fg="white", bg="black",
                               justify='left', font=("Arial", 8))
    else:
        init_button = myButton(init_row_frame, text='START HERE and PASTE then\n wait for temp init complete', command=grab_init, fg="white", bg="black",
                               wraplength=wrap_length, justify='left', font=("Arial", 8))
    open_plink_button = myButton(init_row_frame, text='Open\nPlink', command=open_plink_window, fg='#F5DEB3', bg='#2F4F4F',
                                 activeforeground='wheat', activebackground='#8BA88B', font=("Arial", 8))
    init = tk.StringVar(master, init_val)
    init_label = tk.Label(option_panel_ctr, text='init & clear:', font=label_font_gentle)
    if platform.system() == 'Linux':
        paste_label = tk.Label(option_panel_right, text='ctrl-shift-ins to paste', font=label_font_gentle)
        cmd_label = tk.Label(option_panel_ctr, text=init.get(), font=label_font_gentle)
        init_label.pack(padx=5, pady=5)
    elif platform.system() == 'Darwin':
        paste_label = tk.Label(option_panel_right, text='ctrl-shift-V to paste', font=label_font_gentle)
        cmd_label = tk.Label(option_panel_ctr, text=init.get(), font=label_font_gentle)
        init_label.pack(padx=5, pady=5)
    else:
        paste_label = tk.Label(option_panel_right, text='right-click to paste', font=label_font_gentle)
        cmd_label = tk.Label(option_panel_ctr, text=init.get(), font=label_font_gentle)
        init_label.pack(padx=5, pady=5)
    init_row_frame.pack(padx=5, pady=5)
    init_button.pack(side='left', padx=2, pady=2)
    open_plink_button.pack(side='left', padx=2, pady=2)
    paste_label.pack(padx=5, pady=5)
    cmd_label.pack(padx=5, pady=5)

    # start row
    start = tk.StringVar(master, '')
    start_label = tk.Label(option_panel_left, text='copy start:', font=label_font_gentle)
    start_label.pack(padx=5, pady=5, expand=True, fill='x')
    auto_group = tk.Frame(option_panel_right, relief='groove', bd=2, bg=bg_color)
    auto_group.pack(padx=2, pady=2)
    auto_buttons_row = tk.Frame(auto_group, bg=bg_color)
    if platform.system() == 'Darwin':
        start_button = myButton(option_panel_ctr, text='', command=grab_start, fg="purple", bg=bg_color,
                                justify='left', font=butt_font)
        run_sim_all_button = myButton(auto_buttons_row, text='RunSimAll', command=run_sim_all_batch, fg="blue", bg=bg_color,
                                      justify='left', font=butt_font)
        run_ver_button = myButton(auto_buttons_row, text='RunVer', command=compare_run_ver_batch, fg="blue", bg=bg_color,
                                  justify='left', font=butt_font)
    else:
        start_button = myButton(option_panel_ctr, text='', command=grab_start, fg='#00ff00', bg='black', wraplength=wrap_length,
                                justify='left', font=butt_font)
        run_sim_all_button = myButton(auto_buttons_row, text='RunSimAll', command=run_sim_all_batch, fg="blue", bg=bg_color, wraplength=wrap_length,
                                      justify='left', font=butt_font)
        run_ver_button = myButton(auto_buttons_row, text='RunVer', command=compare_run_ver_batch, fg="blue", bg=bg_color, wraplength=wrap_length,
                                  justify='left', font=butt_font)
    start_button.pack(padx=5, pady=5, expand=True, fill='both')
    auto_button = myButton(auto_buttons_row, text='AUTO', command=grab_auto, fg="blue", bg=bg_color,
                           justify='left', font=butt_font)
    auto_buttons_row.pack(fill='x')
    auto_button.pack(side='left', padx=5, pady=5)
    run_sim_all_button.pack(side='left', padx=5, pady=5)
    run_ver_button.pack(side='left', padx=5, pady=5)
    tk.Label(auto_group, text='Checkboxes don\'t matter', font=label_font_gentle, bg=bg_color).pack(pady=(0, 2))
    auto_plink_path_var = tk.StringVar(master, str(Path(plink_test_csv_path.get()).parent / 'auto_plink.csv'))
    auto_plink_entry = tk.Entry(auto_group, textvariable=auto_plink_path_var, state='readonly',
                                font=label_font_gentle, readonlybackground=bg_color, relief='sunken')
    auto_plink_entry.pack(padx=5, pady=(0, 4), fill='x')
    documentation = tk.BooleanVar(master, cf['others'].get('documentation', 'False') == 'True')
    documentation_button = tk.Checkbutton(option_panel_right, text='Documentation', variable=documentation,
                                          onvalue=True, offvalue=False)
    documentation_button.pack(pady=2, fill='x')
    timer_val = tk.IntVar(master, 0)

    # macro panel
    macro_sep_panel = tk.Frame(master)
    macro_sep_panel.pack(expand=True, fill='x')
    tk.Label(macro_sep_panel, text=' ', font=("Courier", 2), bg='darkgray').pack(expand=True, fill='x')
    macro_panel = tk.Frame(master)
    macro_panel.pack(expand=True, fill='both')
    macro_panel_left = tk.Frame(macro_panel)
    macro_panel_left.pack(side='left', fill='x')
    macro_panel_ctr = tk.Frame(macro_panel)
    macro_panel_ctr.pack(side='left', expand=True, fill='both')
    macro_panel_right = tk.Frame(macro_panel)
    macro_panel_right.pack(side='left', expand=True, fill='both')

    macro_option = tk.StringVar(master, str(cf['others']['macro']))
    macro_option_show = tk.StringVar(master, str(cf['others']['macro']))

    macro_sel = tk.OptionMenu(macro_panel_left, macro_option, *macro_sel_list)
    macro_sel.config(width=20, font=butt_font)
    macro_sel.pack(padx=5, pady=5)
    macro_option.trace_add('write', handle_macro)
    macro = tk.StringVar(master, '')
    if platform.system() == 'Darwin':
        macro_button = myButton(macro_panel_ctr, text=macro.get(), command=grab_macro, fg="purple", bg=bg_color,
                                justify='left', font=butt_font)
    else:
        macro_button = myButton(macro_panel_ctr, text=macro.get(), command=grab_macro, fg="purple", bg=bg_color, wraplength=wrap_length,
                                justify='left', font=butt_font)
    macro_button.pack(padx=5, pady=5)
    get_time_button = myButton(macro_panel_right, text='grab time copy/paste buffer', command=grab_time,
                               fg="blue", bg=bg_color)
    get_time_button.pack(pady=2)

    # Note panel (hidden unless Documentation checkbox is checked)
    note_container = tk.Frame(master)
    note_sep_panel = tk.Frame(note_container)
    note_sep_panel.pack(expand=True, fill='x')
    tk.Label(note_sep_panel, text=' ', font=("Courier", 2), bg='darkgray').pack(expand=True, fill='x')
    note_panel = tk.Frame(note_container)
    note_panel.pack(expand=True, fill='both')
    note_panel_left = tk.Frame(note_panel)
    note_panel_left.pack(side='left', fill='x')
    note_panel_ctr = tk.Frame(note_panel)
    note_panel_ctr.pack(side='left', expand=True, fill='both')
    note_panel_right = tk.Frame(note_panel)
    note_panel_right.pack(side='left', expand=True, fill='both')
    ev1_label = tk.Label(note_panel_ctr, text='', wraplength=wrap_length_note, justify='left', font=note_font)
    ev1_label.pack(padx=5, pady=5, anchor='w')
    ev2_label = tk.Label(note_panel_ctr, text='', wraplength=wrap_length_note, justify='left', font=note_font)
    ev2_label.pack(padx=5, pady=5, anchor='w')
    ev3_label = tk.Label(note_panel_ctr, text='', wraplength=wrap_length_note, justify='left', font=note_font)
    ev3_label.pack(padx=5, pady=5, anchor='w')
    ev4_label = tk.Label(note_panel_ctr, text='', wraplength=wrap_length_note, justify='left', font=note_font)
    ev4_label.pack(padx=5, pady=5, anchor='w')

    def toggle_documentation():
        cf['others']['documentation'] = str(documentation.get())
        cf.save_to_file()
        if documentation.get():
            note_container.pack(before=sav_panel, expand=True, fill='both')
        else:
            note_container.pack_forget()

    # Save row
    sav_panel = tk.Frame(master)
    sav_panel.pack(expand=True, fill='both')
    save_data_label = tk.Label(sav_panel, text='save data:', font=label_font_gentle)
    save_data_label.pack(side='left', padx=5, pady=5)
    save_data_button = myButton(sav_panel, text='save data', command=save_data, fg="red", bg=bg_color,
                                wraplength=wrap_length, justify='left', font=butt_font_large)
    save_data_button.pack(side='left', padx=5, pady=5)


    save_progress_label = tk.Label(sav_panel, text='          ', font=label_font_gentle)
    save_progress_label.pack(side='left', padx=5, pady=5)
    save_progress_button = myButton(sav_panel, text='save progress', command=save_progress, fg="black", bg=bg_color,
                                    wraplength=wrap_length, justify='left')
    save_progress_button.pack(side='left', padx=5, pady=5)


    terse_str = cf['others']['terse']
    if terse_str == 'True':
        terse = tk.BooleanVar(master, True)
    else:
        terse = tk.BooleanVar(master, False)
    terse_button = tk.Checkbutton(sav_panel, text='terse plots', variable=terse, onvalue=True, offvalue=False)
    terse_button.pack(side='left', pady=2, fill='x')
    terse.trace_add('write', handle_terse)


    strict_overplot_str = cf['others']['strict_overplot']
    if strict_overplot_str == 'True':
        strict_overplot = tk.BooleanVar(master, True)
    else:
        strict_overplot = tk.BooleanVar(master, False)
    strict_overplot_button = tk.Checkbutton(sav_panel, text='strict_overplot plots', variable=strict_overplot, onvalue=True, offvalue=False)
    strict_overplot_button.pack(side='left', pady=2, fill='x')
    strict_overplot.trace_add('write', handle_strict_overplot)

    hardcopy_str = cf['others'].get('hardcopy', 'False')
    hardcopy = tk.BooleanVar(master, hardcopy_str == 'True')
    hardcopy_button = tk.Checkbutton(sav_panel, text='hardcopy', variable=hardcopy, onvalue=True, offvalue=False)
    hardcopy_button.pack(side='left', pady=2, fill='x')
    hardcopy.trace_add('write', handle_hardcopy)
    documentation.trace_add('write', lambda *_: toggle_documentation())
    toggle_documentation()  # apply initial state from config

    clear_data_button = myButton(sav_panel, text='clear', command=clear_data_verbose, fg="red", bg=bg_color,
                                 wraplength=wrap_length, justify='right')
    clear_data_button.pack(side='right', padx=5, pady=5)
    save_data_as_button = myButton(sav_panel, text='save as', command=save_data_as, fg="red", bg=bg_color,
                                   wraplength=wrap_length, justify='left')
    save_data_as_button.pack(side='right', padx=5, pady=5)


    # Run panel
    mod_in_app = tk.IntVar(master, int(cf['others']['mod_in_app']))
    run_sep_panel = tk.Frame(master)
    run_sep_panel.pack(expand=True, fill='x')
    tk.Label(run_sep_panel, text=' ', font=("Courier", 2), bg='darkgray').pack(expand=True, fill='x')
    run_panel = tk.Frame(master)
    run_panel.pack(expand=True, fill='x')
    tk.Label(run_panel, text='------->', font=("Courier", 8), bg='lightgreen').pack(side='left')
    if platform.system() == 'Darwin':
        run_x_button = myButton(run_panel, text=' Compare ', command=compare_run, fg="green", bg=bg_color,
                                  justify='left', font=butt_font_large)
        hist_hist_button = myButton(run_panel, text='Compare Hist Hist', command=compare_hist_hist_run, fg="green",
                                    bg=bg_color, justify='left', font=butt_font_large)
        hist_sim_button = myButton(run_panel, text=' Compare ', command=compare_hist_to_sim, fg="green", bg=bg_color,
                                   justify='left', font=butt_font_large)
        run_sim_hist_button = myButton(run_panel, text=' Compare ', command=compare_run_to_hist, fg="green", bg=bg_color,
                                       justify='left', font=butt_font_large)
    else:
        run_x_button = myButton(run_panel, text=' Compare ', command=compare_run, fg="green", bg=bg_color,
                              wraplength=wrap_length, justify='left', font=butt_font_large)
        hist_hist_button = myButton(run_panel, text='Compare Hist Hist', command=compare_hist_hist_run, fg="green",
                                    bg=bg_color, justify='left', font=butt_font_large)
        hist_sim_button = myButton(run_panel, text=' Compare ', command=compare_hist_to_sim, fg="green", bg=bg_color,
                                   justify='left', font=butt_font_large)
        run_sim_hist_button = myButton(run_panel, text=' Compare ', command=compare_run_to_hist, fg="green", bg=bg_color,
                                   justify='left', font=butt_font_large)
    mod_in_app_button = myButton(run_panel, text=mod_in_app.get(), command=enter_mod_in_app, fg="green", bg=bg_color)
    run_x_button.pack(side='left', padx=5, pady=5)
    hist_hist_button.pack(side='left', padx=5, pady=5)
    mod_in_app_button.pack(side='right', padx=5, pady=5)
    hist_sim_button.pack(side='right', padx=5, pady=5)
    run_sim_hist_button.pack(side='right', padx=5, pady=5)

    # Compare panel
    compare_sep_panel = tk.Frame(master)
    compare_sep_panel.pack(expand=True, fill='x')
    tk.Label(compare_sep_panel, text=' ', font=("Courier", 2), bg='darkgray').pack(expand=True, fill='x')
    tk.ttk.Separator(compare_sep_panel, orient='horizontal').pack(pady=5, side='top')
    compare_panel = tk.Frame(master)
    compare_panel.pack(expand=True, fill='x')
    choose_label = tk.Label(compare_panel, text='choose existing files:')
    choose_label.pack(side='left', padx=5, pady=5)
    run_sim_choose_button = myButton(compare_panel, text='Compare Run Sim Choose', command=compare_run_sim_choose,
                                     fg="blue", bg=bg_color, wraplength=wrap_length, justify='left', font=butt_font)
    run_sim_choose_button.pack(side='left', padx=5, pady=5)
    run_run_choose_button = myButton(compare_panel, text='Compare Run Run Choose', command=compare_run_run_choose,
                                     fg="blue", bg=bg_color, wraplength=wrap_length, justify='left', font=butt_font)
    run_run_choose_button.pack(side='left', padx=5, pady=5)
    run_sim_choose_button = myButton(compare_panel, text='Compare Hist Sim Choose', command=compare_hist_sim_choose,
                                     fg="blue", bg=bg_color, wraplength=wrap_length, justify='left', font=butt_font)
    run_sim_choose_button.pack(side='left', padx=5, pady=5)
    hist_hist_choose_button = myButton(compare_panel, text='Compare Hist Hist Choose', command=compare_hist_hist_choose,
                                       fg="blue", bg=bg_color, wraplength=wrap_length, justify='left', font=butt_font)
    hist_hist_choose_button.pack(side='left', padx=5, pady=5)

    # Begin
    handle_test_unit()
    handle_run_unit()
    handle_test_battery()
    handle_run_battery()
    handle_modeling()
    handle_terse()
    handle_macro()
    handle_option()
    master.mainloop()
