#! /bin/sh
# noinspection PySingleQuotedDocstring
"exec" "`dirname $0`/venv/bin/python3" "$0" "$@"
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
import shutil
import pyperclip
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
    default_dict,
    empty_file,
    ExRoot,
    lookup,
    macro_lookup,
    macro_sel_list,
    plat,
    plink_connection,
    register_last_task,
    run_previous_task,
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
        if not Path(self.version_path).is_dir():
            tk.messagebox.showerror(title="Error",
                                    message=self.version_path + " unavailable. Abort opening\nTurn on Drive & refresh" +
                                                                " dataReduction Folder.")
        else:
            try:
                os.makedirs(self.version_path, exist_ok=True)
            except OSError:
                tk.messagebox.showerror(title="Error", message="check " + self.version_path + " available")
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
    if Path(putty_test_csv_path.get()).is_file():
        enter_size = putty_size()  # bytes
        time.sleep(1.)
        wait_size = putty_size()  # bytes
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
            if not save_putty():
                if not silent:
                    tkinter.messagebox.showwarning(message="putty may be open already")
                else:
                    update_data_buttons()
    else:
        if not silent:
            print('putty test file non-existent or too small (<64 bytes) probably already done')
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
                print('GUI_TestSOC compare_hist_hist_choose:  Ref', ref_basename, ref_key)
                print('GUI_TestSOC compare_hist_hist_choose:  Test', test_basename, test_key)
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
                         terse=terse.get(), strict_overplot=strict_overplot.get())
    else:
        print('not possible')


def compare_run():
    register_last_task(compare_run)
    if not Test.key_exists_in_file:
        tkinter.messagebox.showwarning(message="Test Key '" + Test.key + "' does not exist in " + Test.file_txt)
        return
    update_data_buttons()
    if modeling.get():
        print('compare_run_sim.  save_pdf_path', str(PurePosixPath(Test.version_path) / 'figures'))
        compare_run_sim(data_file=Test.file_path, unit_key=Test.key, strict_overplot=strict_overplot.get(),
                        terse=terse.get())
    else:
        if not Ref.key_exists_in_file:
            tkinter.messagebox.showwarning(message="Ref Key '" + Ref.key + "' does not exist in " + Ref.file_txt)
            return
        print('GUI_TestSOC compare_run:  Ref', Ref.file_path, Ref.key)
        print('GUI_TestSOC compare_run:  Test', Test.file_path, Test.key)
        keys = [(Ref.file_txt, Ref.key), (Test.file_txt, Test.key)]
        compare_run_run(keys=keys, data_file_folder_run=Ref.version_path, data_file_folder_test=Test.version_path,
                        terse=terse.get())



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
                      dt_resample=answer, terse=terse.get())


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
def compare_run_sim_choose():
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
                        terse=terse.get())
            else:
                tk.messagebox.showerror(message='key not found in' + testpath)
        update_data_buttons()


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
    init_button.config(bg='darkslategray', activebackground='darkslategray', fg='wheat', activeforeground='wheat')
    start_button.config(bg='darkslategray', activebackground='darkslategray', fg='wheat', activeforeground='wheat')
    get_time_button.config(bg=bg_color, activebackground=bg_color, fg='black', activeforeground='purple')


def grab_init():
    register_last_task(grab_init)
    # Grab command to update time in EEPROM
    try:
        current_ut = 'UT' + str(int(time.time())) + ';'
    except AttributeError:
        current_ut = ''
        print(f"current_ut blank ***No Internet??")
    init_command = init.get() + current_ut
    print(f"Init command to paste: {init_command}")
    add_to_clip_board(init_command)
    # Grab the rest
    grab_all_nominal()
    init_button.config(bg='yellow', activebackground='yellow', fg='black', activeforeground='black')
    clear_data_silent()
    print('cleared putty data file')
    Test.create_file_path_and_key()
    Test.update_key_label()
    start_putty()


def monitor_putty_done():
    if Path(putty_test_csv_path.get()).is_file():
        try:
            with open(putty_test_csv_path.get(), 'rb') as f:
                f.seek(0, 2)
                size = f.tell()
                # Read last 1024 bytes to check for ***DONE***
                f.seek(max(0, size - 1024))
                last_data = f.read().decode('utf-8', errors='ignore')
                if '***DONE***' in last_data:
                    print(f"***DONE*** detected in {putty_test_csv_path.get()}")
                    save_data()
                    tk.messagebox.showinfo(title='Done ' + start_button.cget('text'), message='Run Complete')
                    return
        except Exception as e:
            print(f"Error monitoring putty file: {e}")
    master.after(1000, monitor_putty_done)


def grab_start():
    global grab_start_time
    grab_start_time = time.time()
    register_last_task(grab_start)
    add_to_clip_board(start.get())
    grab_all_nominal()
    save_data_button.config(bg=bg_color, activebackground=bg_color, fg='black', activeforeground='black',
                            text='save data')
    save_data_as_button.config(bg=bg_color, activebackground=bg_color, fg='black', activeforeground='black',
                               text='save data as')
    grab_all_nominal()
    start_button.config(bg='yellow', activebackground='yellow', fg='black', activeforeground='black')
    start_timer()
    monitor_putty_done()


def grab_all_nominal():
    macro_button.config(bg=bg_color, activebackground='black', fg='black', activeforeground='white')
    init_button.config(bg='darkslategray', activebackground='black', fg='wheat', activeforeground='wheat')
    start_button.config(bg='darkslategray', activebackground='darkslategray', fg='wheat', activeforeground='wheat')
    get_time_button.config(bg=bg_color, activebackground='black', fg='black', activeforeground='white')


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

    # Check if this is what you want to do
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

    # Check if this is what you want to do
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
    start_button.config(bg='darkslategray', activebackground='darkslategray', fg='wheat', activeforeground='wheat')
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


def kill_putty(sys_=None, silent=True):
    command = ''
    if sys_ == 'Linux':
        command = 'pkill -e putty'
    elif sys_ == 'Windows':
        command = 'taskkill /f /im putty.exe'
    elif sys_ == 'Darwin':
        command = 'pkill putty'
    else:
        print(f"kill_putty: SYS = {sys_} unknown")
    if not silent:
        print(command + '\n')
        print(Colors.bg.brightblack, Colors.fg.wheat)
        result = run_shell_cmd(command, silent=silent)
        print(Colors.reset)
        print(command + '\n')
        if result == -1:
            print(Colors.fg.blue, 'failed.', Colors.reset)
            return None, False
    else:
        result = run_shell_cmd(command, silent=silent)
    return result


def look_putty(sys_=None, silent=True):
    if sys_ == 'Linux':
        try:
            output = subprocess.check_output(['pgrep', '-f', 'putty']).decode('ascii')
            return len(output.strip()) > 0
        except subprocess.CalledProcessError:
            return False
    elif sys_ == 'Windows':
        try:
            output = subprocess.check_output(['tasklist', '/FI', 'IMAGENAME eq putty.exe', '/NH']).decode('ascii')
            return 'putty.exe' in output.lower()
        except subprocess.CalledProcessError:
            return False
    elif sys_ == 'Darwin':
        try:
            output = subprocess.check_output(['pgrep', '-f', 'putty']).decode('ascii')
            return len(output.strip()) > 0
        except subprocess.CalledProcessError:
            return False
    else:
        print(f"look_putty: SYS = {sys_} unknown")
        return False


def lookup_macro():
    dum_, macro_val, ev_val = macro_lookup.get(macro_option.get())
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
    dawdle_val_, start_val, ev_val = lookup.get(option.get())
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


def putty_size():
    if Path(putty_test_csv_path.get()).is_file():
        enter_size = Path(putty_test_csv_path.get()).stat().st_size  # bytes
    else:
        enter_size = 0
    return enter_size


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


def save_data():
    global timer, grab_start_time
    print(f"save_data: {putty_test_csv_path.get()=}")
    if size_of(putty_test_csv_path.get()) > 64:  # bytes
        # For custom option, redefine Test.file_path if requested
        new_file_txt = None
        if option.get() == 'custom':
            new_file_txt = tk.simpledialog.askstring(title=__file__, prompt="custom file name string:")
            if new_file_txt is not None:
                Test.create_file_path_and_key(name_override=new_file_txt)
                Test.label.config(text=Test.file_txt)
                print('Test.file_path', Test.file_path)
        if Path(Test.file_path).is_file() and Path(Test.file_path).stat().st_size > 0:  # bytes
            if auto_overwrite.get():
                print('auto over-write enabled')
            else:
                confirmation = tk.messagebox.askyesno('query overwrite', 'File exists:  overwrite?')
                if not confirmation:
                    print('skipped overwrite')
                    tkinter.messagebox.showwarning(message='retained ' + Test.file_path)
                    return
        save_data_button.config(bg='yellow', activebackground='yellow', fg='black', activeforeground='black',
                                text='data saving')
        tksleep(0.1)
        if grab_start_time is not None:
            elapsed_s = time.time() - grab_start_time
            print(f"Run elapsed time: {elapsed_s:.1f} s")
            with open(putty_test_csv_path.get(), 'a') as _f:
                _f.write(f"elapsed_s,{elapsed_s:.1f}\n")
            grab_start_time = None
        copy_clean(putty_test_csv_path.get(), Test.file_path)
        print('copied ', putty_test_csv_path.get(), '\nto\n', Test.file_path)
        if timer is not None:
            timer.close()
            timer = None
        save_data_button.config(bg='green', activebackground='green', fg='red', activeforeground='red',
                                text='data saved')
        empty_file(putty_test_csv_path.get())
        print('updating Test file label')
        Test.create_file_path_and_key(name_override=new_file_txt)
        if auto_overwrite.get():
            print('auto over-write triggering comparison')
            compare_run()
    else:
        print('putty test file non-existent or too small (<64 bytes) probably already done')
        tkinter.messagebox.showwarning(message="Nothing to save")
    start_button.config(bg='darkslategray', activebackground='darkslategray', fg='wheat', activeforeground='wheat')


def save_data_as():
    global timer
    if size_of(putty_test_csv_path.get()) > 512:  # bytes
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
        copy_clean(putty_test_csv_path.get(), Test.file_path)
        print('copied ', putty_test_csv_path.get(), '\nto\n', Test.file_path)
        if timer is not None:
            timer.close()
            timer = None
        save_data_as_button.config(bg='green', activebackground='green', fg='red', activeforeground='red',
                                   text='data saved as')
        empty_file(putty_test_csv_path.get())
        print('updating Test file label')
        Test.create_file_path_and_key(name_override=new_file_txt)
    else:
        print('putty test file is too small (<512 bytes) probably already done')
        tkinter.messagebox.showwarning(message="Nothing to save")
    start_button.config(bg='darkslategray', activebackground='darkslategray', fg='wheat', activeforeground='wheat')


def save_progress():
    global timer
    if size_of(putty_test_csv_path.get()) > 64:  # bytes
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
        copy_clean(putty_test_csv_path.get(), Test.file_path)
        if timer is not None:
            timer.close()
            timer = None
        save_progress_button.config(bg=bg_color, activebackground=bg_color, fg='black', activeforeground='black',
                                    text='save_progress')
        print('copied ', putty_test_csv_path.get(), '\nto\n', Test.file_path)
        print('updating Test file label')
        Test.create_file_path_and_key(name_override=new_file_txt)
        tkinter.messagebox.showwarning(message="Progress saved")
        update_data_buttons()
    else:
        print('putty test file non-existent or too small (<64 bytes) probably already done')
        tkinter.messagebox.showwarning(message="Nothing to save")


def save_putty():
    m_str = datetime.datetime.fromtimestamp(Path(putty_test_csv_path.get()).stat().st_mtime).strftime("%Y-%m-%dT%H-%M-%S").replace(' ', 'T')
    putty_test_sav_path = tk.StringVar(master, str(PurePosixPath(path_to_temp.get()) / ('putty_' + m_str + '.csv')))
    print(f"GUI_TestSOC(save_putty):\n{putty_test_csv_path.get()=}\n{putty_test_sav_path.get()=}\n")
    try:
        shutil.copyfile(putty_test_csv_path.get(), putty_test_sav_path.get())
        print('wrote', putty_test_sav_path.get())
        empty_file(putty_test_csv_path.get())
        return True
    except PermissionError:
        print('putty holding file open')
        return False


def start_putty():
    lookup_test()
    if look_putty(platform.system()):
        print("PuTTY already open.  Skipping restart.")
        return

    enter_size = putty_size()
    if enter_size >= 64:
        if not save_putty():
            tkinter.messagebox.showwarning(message="putty may be open already")
        enter_size = putty_size()

    if enter_size < 64:
        kill_putty(platform.system())
        print(f'restarting putty   putty -load {test_filename.get()}')
        subprocess.Popen(['putty', '-title', 'putty-terminal-server', '-load', test_filename.get()],
                         stdin=subprocess.PIPE, bufsize=1, universal_newlines=True)


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
    start_button.config(bg='darkslategray', activebackground='darkslategray', fg='wheat', activeforeground='wheat')


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
    master = tk.Tk(className='GUI_TestSOC')
    print("master created")
    master.title('State of Charge')
    master.wm_minsize(width=min_width, height=main_height)
    timer = None
    grab_start_time = None
    print("creating Ref")
    Ref = Exec(cf, 'ref', path_disp_len_=folder_reveal)
    print("Ref created")
    print("creating Test")
    Test = Exec(cf, 'test', path_disp_len_=folder_reveal)
    print("Test created")
    if platform.system() == 'Linux':
        putty_test_csv_path = tk.StringVar(master, '/home/daveg/.local/putty_test.csv')
        path_to_temp = tk.StringVar(master, '/home/daveg/.local')
    elif platform.system() == 'Darwin':
        putty_test_csv_path = tk.StringVar(master, '/Users/daveg/.local/putty_test.csv')
        path_to_temp = tk.StringVar(master, '/Users/daveg/.local')
    else:
        local_app_data_ = os.getenv('LOCALAPPDATA') or str(Path.home() / 'AppData' / 'Local')
        putty_test_csv_path = tk.StringVar(master, str(Path(local_app_data_) / 'Temp' / 'putty_test.csv'))
        path_to_temp = tk.StringVar(master, str(Path(local_app_data_) / 'Temp'))
    print(f"{putty_test_csv_path.get()=}")
    print("loading icon")
    icon_path = str(PurePosixPath(ex_root.script_loc) / 'GUI_TestSOC.png')
    master.iconphoto(False, tk.PhotoImage(file=icon_path))
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

    # Image
    pic_path = str(PurePosixPath(ex_root.script_loc) / 'GUI_TestSOC.png')
    picture = tk.PhotoImage(file=pic_path).subsample(5, 5)
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
    if platform.system() == 'Darwin':
        init_button = myButton(option_panel_ctr, text='START HERE and PASTE then\n wait for temp init complete', command=grab_init, fg='wheat', bg='darkslategray',
                               justify='left', font=("Arial", 8))
    else:
        init_button = myButton(option_panel_ctr, text='START HERE and PASTE then\n wait for temp init complete', command=grab_init, fg='wheat', bg='darkslategray',
                               wraplength=wrap_length, justify='left', font=("Arial", 8))
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
    init_button.pack(padx=5, pady=5)
    paste_label.pack(padx=5, pady=5)
    cmd_label.pack(padx=5, pady=5)

    # start row
    start = tk.StringVar(master, '')
    start_label = tk.Label(option_panel_left, text='copy start:', font=label_font_gentle)
    start_label.pack(padx=5, pady=5, expand=True, fill='x')
    if platform.system() == 'Darwin':
        start_button = myButton(option_panel_ctr, text='', command=grab_start, fg='wheat', bg='darkslategray',
                                justify='left', font=butt_font)
        prev_button = myButton(option_panel_right, text='Run Prev', command=run_previous_task, fg="blue", bg=bg_color,
                                justify='left', font=butt_font)
    else:
        start_button = myButton(option_panel_ctr, text='', command=grab_start, fg='wheat', bg='darkslategray', wraplength=wrap_length,
                                justify='left', font=butt_font)
        prev_button = myButton(option_panel_right, text='Run Prev', command=run_previous_task, fg="blue", bg=bg_color, wraplength=wrap_length,
                                justify='left', font=butt_font)
    start_button.pack(padx=5, pady=5, expand=True, fill='both')
    prev_button.pack(padx=5, pady=5)
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

    # Note panel
    note_sep_panel = tk.Frame(master)
    note_sep_panel.pack(expand=True, fill='x')
    tk.Label(note_sep_panel, text=' ', font=("Courier", 2), bg='darkgray').pack(expand=True, fill='x')
    note_panel = tk.Frame(master)
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
