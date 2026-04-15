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

sys.stdout.write("\033]0;SOC\007")
sys.stdout.flush()

plat = sys.platform
if plat == 'linux':
    default_dr = '/home/daveg/gdrive/GitHubArchive/SOC_Particle/dataReduction'
elif plat == 'darwin':
    default_dr = '/Users/daveg/Library/CloudStorage/GoogleDrive-davegutz2006@gmail.com/My Drive/GitHubArchive/SOC_Particle/dataReduction'
else:
    default_dr = 'G:/My Drive/GitHubArchive/SOC_Particle/dataReduction'

# Tee stdout/stderr to a log file so Console.app shows output when launched as a .app bundle
_log_dir = os.path.expanduser("~/Library/Logs") if plat == 'darwin' else os.path.expanduser("~")
os.makedirs(_log_dir, exist_ok=True)
_log_file = open(os.path.join(_log_dir, "GUI_TestSOC.log"), 'a', buffering=1)


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)

    def flush(self):
        for s in self._streams:
            s.flush()


last_task = None
last_task_args = ()
last_task_kwargs = {}
plink_pid = None


def register_last_task(func, *args, **kwargs):
    global last_task, last_task_args, last_task_kwargs
    last_task = func
    last_task_args = args
    last_task_kwargs = kwargs


def run_previous_task():
    global last_task, last_task_args, last_task_kwargs
    if last_task is not None:
        print(f"Running previous task: {last_task.__name__}")
        last_task(*last_task_args, **last_task_kwargs)
    else:
        print("No previous task to run")


sys.stdout = _Tee(sys.__stdout__, _log_file)
sys.stderr = _Tee(sys.__stderr__, _log_file)

# Configuration for entire folder selection read with filepaths
def_dict = {
    'test': {
        "version": "g20240331",
        "unit": "pro2p2",
        "battery": "bb",
        'dataReduction_folder': default_dr,
    },
    'ref': {
        "version": "g20240331",
        "unit": "pro0p",
        "battery": "bb",
        'dataReduction_folder': default_dr,
    },
    'others': {
        "option": "custom",
        'macro': 'end_early',
        'mod_in_app': "247",
        'modeling': True,
        'strict_overplot':True,
        'terse': True,
        'auto_overwrite': False,
    },
    }

# Transient string
unit_list = [
    'pro0p', 'pro1a', 'pro2p2', 'pro2p2_hi_lo', 'pro3p2', 'pro3p2_hi_lo', 'pro4p2', 'soc0p', 'soc1a', 'soc2p2_hi_lo',
    'soc3p2_hi_lo', 'soc4p2_hi_lo',
    ]
battery_list = ['bb', 'chg']
sel_list = [
    'custom', 'init1', 'saveAdjusts', 'ampHiEmptFail', 'ampHiFail', 'noaHiFail', 'rapidTweakRegression', 'allInBB',
    'allProto', 'pulseSoft', 'pulseHard', 'rapidTweakRegressionH0', 'offLowSoc', 'offSitHysBmsBB',
    'offSitHysBmsCHG', 'triTweakDisch', 'ampHiFailFf', 'ampLoFail', 'ampLoFullFail', 'noaLoFail', 'noaLoFullFail', 'ampHiFailNoise', 'noaHiFailNoise',
    'rapidTweakRegression40C', 'slowTweakRegression', 'satSitBB', 'satSitCHG',
    ]
sel_list1 = [
    'flatSitHys', 'offSitHysBmsNoiseBB', 'offSitHysBmsNoiseCHG', 'ampHiFailSlow',
    'noaHiFailSlow', 'noaHiFailSlower', 'noaHiFailSlowest', 'vHiFail', 'vHiFailNoise', 'vHiFailH', 'vHiFailFf',
    'pulseHard', 'tLoFailHdwe', 'DvMon', 'DvSim', 'faultParade', 'stepDown', 'stepUp', 'zero_with_pc',
    ]
macro_sel_list = [
    'end_early', 'hdwNoVbPcMidInit', 'modHalfInit', 'modEmptInitBB', 'modEmptInitCHG',
    'noisePackage', 'silentPackage', 'quiet', 'cleanup', 'tempCleanup', 'tranPrep', 'synced_slow', 'slow',
    'slowTwitchDef', 'fastTwitchDef', 'c06', 'd06', 'c08', 'd05', 'd08', 'c10', 'd10', 'c18', 'd18', 'c50', 'cm50', 'c00',
    'dv0', 'twitch', 'time_stamp', 's00', 'sd50', 'sc50', 'zeroPrepHdweNoVb', 'zero_set_hdwe_no_Vb',
    'noaHiFail', 'noaHiFailNoise',
    ]

# Macro
satInit = 'Dh;*W;*vv0;*XS;*Ca1;BZ;Ff0;DP1;HR;Rf;'
hdwNoVbPcMidInit = 'vv0;Xm2;Ca0.50;BZ;Ff0;W20;DP1;HR;Rf;'
modFullInit = 'vv0;Xm247;Ca0.93;BZ;Ff0;DP1;HR;Rf;'  # kickers off 0.94
modLoInit = 'vv0;Xm247;Ca0.17;BZ;Ff0;DP1;HR;Rf;'
modHalfInit = 'vv0;Xm247;Ca0.50;BZ;Ff0;DP1;HR;Rf;'
modHalfInitNoCc = 'vv0;Xm247;Ca0.50;BZ;Ff0;DP1;HR;Rf;'
modEmptInitBB = 'vv0;Xm247;Ca0.090;BZ;Ff0;DP1;HR;Rf;'
modEmptInitCHG = 'vv0;Xm247;Ca-0.004;BZ;Ff0;DP1;HR;Rf;'
modEmptInitGen = 'vv0;Xm247;Ca0.17;BZ;Ff0;DP1;HR;Rf;'
noisePackage = 'DT.05;DV0.3;DM.75;DN6;'
silentPackage = 'DT0;DV0;DM0;DN0;'
synced_slow = 'Dr400;D>400;Dq400;ED1;DP1;'
synced_slow_pulse = 'Dr800;D>800;Dq800;ED1;DP1;'
slow = synced_slow
quiet = 'vv0;Dr;Dq;DP;D>;Dh;'
quietwait = '<vv0;Dr;DP;D>;Dh;'
cleanup = 'Hd;Pf;<HR;<Rf;'
tempCleanup = 'Rf; '
time_stamp = 'XY;'
zeroPrepHdweNoVb = 'HR;Dh1000;W34;Fi2;Fo2;Rs;W34;'
zero_set_hdwe_no_Vb = 'vv0;Xm2;Ca0.50;W20;BZ;Ff1;DP1;HR;Fi2;Fo2;Rf;vv99;W1;<Xm2;'
tranPrep = 'HR;Dh1000;W2;Rs;W48;vv4;W17;'
slowTranPrep = 'HR;vv4;W2;Rs;' + slow + 'W5;'
slowTwitchDef = 'Rb;Rf;Sh0;Xts;Xf0.004;Mm1000;Mn-1000;Nm1000;Nn-1000;XW10000;XT10;XC2;'
fastTwitchDef = 'Rb;Rf;Xts;Xf0.002;XW10000;XT10;XC1;'
c18 = time_stamp + 'Dm18;Dn0.0001;'  # 0.0001 helps saturation logic behave correctly in a quiet simulation
d18 = time_stamp + 'Dn18;Dm0.0001;'  # 0.0001 helps saturation logic behave correctly in a quiet simulation
c06 = time_stamp + 'Dm6;Dn0.0001;'  # 0.0001 helps saturation logic behave correctly in a quiet simulation
d06 = time_stamp + 'Dn6;Dm0.0001;'  # 0.0001 helps saturation logic behave correctly in a quiet simulation
c08 = time_stamp + 'Dm8;Dn0.0001;'  # 0.0001 helps saturation logic behave correctly in a quiet simulation
d05 = time_stamp + 'Dn5;Dm0.0001;'  # 0.0001 helps saturation logic behave correctly in a quiet simulation
d08 = time_stamp + 'Dn8;Dm0.0001;'  # 0.0001 helps saturation logic behave correctly in a quiet simulation
c10 = time_stamp + 'Dm10;Dn0.0001;'  # 0.0001 helps saturation logic behave correctly in a quiet simulation
d10 = time_stamp + 'Dn10;Dm0.0001;'  # 0.0001 helps saturation logic behave correctly in a quiet simulation
d20 = time_stamp + 'Dn20;Dm0.0001;'  # 0.0001 helps saturation logic behave correctly in a quiet simulation
c50 = time_stamp + 'Dm50;Dn0.0001;'  # 0.0001 helps saturation logic behave correctly in a quiet simulation
d50 = time_stamp + 'Dn50;Dm0.0001;'  # 0.0001 helps saturation logic behave correctly in a quiet simulation
cm50 = time_stamp + 'Dm-50;Dn0.0001;'  # 0.0001 helps saturation logic behave correctly in a quiet simulation
dm50 = time_stamp + 'Dn-50;Dm0.0001;'  # 0.0001 helps saturation logic behave correctly in a quiet simulation50
sc50 = time_stamp + 'DI50;'  # 50 amp discharge
sd50 = time_stamp + 'DI-50;'  # 50 amp discharge
c00 = 'Pf;W2;Dm0;Dn0;Rf;W50;'
dv0 = 'Pf;W2;Dv0;Rf;W50;'
s00 = 'Pf;W2;DI0;Rf;W100;'
twitch = time_stamp + 'XR;'
vm12 = 'Dv-12;'

# Note:  Photon 2 is throughput limited on the Serial buses.  The *tweak* transients are sensitive to differences
# caused by over-runs and slip and set Dr400 before Xp* then resets to Dr100 (nominal).
lookup = {
        'satInit': (22, 'Y;' + quiet + 'cc;Dh;Dr;*W;*vv0;*XS;*Ca1;BZ;Ff0;DP1;<HR;<Rf;<XK;', ('',)),
        'initMid': (22, 'Y;' + quiet + 'cc;Dh1800000;*W;*vv0;*XS;*Ca.5;BZ;Ff0;<HR;<Rf;<XK;', ('',)),
        'saveAdjusts': (60, 'vv4;Dh1000;PR;PV;Pr;Pr;BP2;Pr;BP1;Pr;BS2;Pr;BS1;Pr;Pr;Pr;DA5;Pr;DB-5;Pr;RS;Pr;Dc0.2;Pr;Dc0;DI-10;Pr;DI0;Pr;Dt5;Pr;Dt0;Pr;SA2;Pr;SA1;Pr;SB2;Pr;SB1;Pr;si-1;Pr;RS;Pr;Sk2;Pr;Sk1;Pr;SQ2;Pr;SQ1;Pr;Sq3;Pr;Sq1;Pr;SV1.1;Pr;SV1;Pr;Xb10;Pr;Xb0;Pr;Xa1000;Pr;Xa0;Pr;Xf1;Pr;RS;Pr;Xm10;Pr;RS;Pr;W3;vv0;XQ3;PR;PV;XQ60000;Dh;<XD;', ("For testing out the adjustments and memory", "Read through output and witness set and reset of all", "The DS2482 moderate headroom should not exceed limit printed.  EG 11 of 12 is ok.")),
        'custom': (72, 'XQ60000;<XD;', ("For general purpose data collection", "'save data' will present a choice of file name", "")),
        'allInBB': (1200,
                    slow + 'Dh4000;' +
                    modEmptInitBB + slowTwitchDef + 'Xa-162;' + slowTranPrep + twitch + 'XQ568000;' + 'Xa0;' + tempCleanup +  # offSitHysBmsBB
                    'Xm247;Ca0.9962;' + fastTwitchDef + 'Xa17;' + slowTranPrep + 'XR;XQ600000;' + 'Xa0;' +  # satSitBB
                    quiet + cleanup + '<XD;',
                    ('All the best transients BB', "Must have same 'vv*' throughout", "")),
        'ampHiEmptFail': (153, modLoInit + tranPrep + c50 + 'XQ25000;' + c00 + quiet + cleanup + '<XD;', ("Inject 50A into amp.  Should detect and switch amp current failure", "'diff' will be displayed. After a bit more, current display will change to 0.", "To evaluate plots, start looking at 'Ult 1'. Fault record (frozen). Will see 'diff' flashing on display soon after fault cleared automatically (lost redundancy).  Also will see verification imbedded model respond to the bad current signal by elevating vb, an effect that won't appear in data from app.", "Loss of ibm set 'accy' because loss of most accurate sensor.")),
        'ampHiFail': (153, modHalfInit + tranPrep + c50 + 'XQ25000;' + c00 + quiet + cleanup + '<XD;', ("Inject 50A into amp.  Should detect and switch amp current failure", "'diff' will be displayed. After a bit more, current display will change to 0.", "To evaluate plots, start looking at 'Ult 1'. Fault record (frozen). Will see 'diff' flashing on display soon after fault cleared automatically (lost redundancy).  Also will see verification imbedded model respond to the bad current signal by elevating vb, an effect that won't appear in data from app.", "Loss of ibm set 'accy' because loss of most accurate sensor.")),
        'noaHiFail': (153, modHalfInit + tranPrep + d50 + 'XQ25000;' + c00 + quiet + cleanup + '<XD;', ("Inject 50A into amp. With ib_diff only nothing changes then should isolate to the noa by wrap and choose amp.", "'diff' will be displayed then ib_fail due to wrap of noa", "To evaluate plots, start looking at 'Ult 1'. Fault record (frozen).", "Loss of ib set 'accy' because loss of current sensing at high currents.")),
        'rapidTweakRegression': (262, slow + 'Rs;W8;Xp10;' + quiet + cleanup + '<XD;', ('Should run three very large current discharge/recharge cycles without latched fail', 'Best test for seeing time skews and checking fault logic for false trips', 'Occasional jumps in ib_sel_stat are normal when pass through 0 A.  And Noa will fault and fail temprorarily')),
        'allProto': (552, modHalfInit + tranPrep + c50 + 'XQ25000;' + c00 + tempCleanup + '  Rs;W4;Xp10;  Rs;W4;Xp13;  ' + modHalfInitNoCc + tranPrep + cm50 + 'XQ50000;' + c00 + quiet + cleanup + '<XD;', ('Proto multi', "Must have same 'vv*' throughout", "No 'HR' either")),
        'pulseSoft': (75, synced_slow_pulse + 'XS;Dm0;Dn0;vv0;Xm255;Ca.5;Pm;W2;Rs;W20;vv4;W10;' + 'Xp7;W10;Pc;' + quiet + cleanup + '<XD;', ("Should generate a very short <10 sec data burst with a current sensor pulse.  Look at plots for good overlay. e_wrap should be nearly flat after a pulse response.", "This is the shortest of all tests.  Also useful for quick check tests.", "")),
        'pulseHard': (75, synced_slow_pulse + 'XS;Dm0;Dn0;vv0;Xm255;Ca.5;Pm;W2;Rs;W20;vv4;W10;' + 'Xp8;W10;Pc;' + quiet + cleanup + '<XD;', ("Should generate a very short <10 sec data burst with a hardware current pulse.  Look at plots for good overlay. e_wrap should be flat.", "This is the shortest of all tests.  Also useful for quick check tests.", "")),
        'rapidTweakRegressionH0': (262, 'Sh0;' + slow + 'Rs;W4;Xp10;Pf;W2;' + quiet + cleanup + '<XD;', ('Should run three very large current discharge/recharge cycles without fault', 'No hysteresis. Best test for seeing time skews and checking fault logic for false trips', 'Tease out cause of e_wrap faults.  e_wrap MUST be flat!', 'Occasional jumps in ib_sel_stat are normal when pass through 0 A')),
        'offLowSoc': (172, modEmptInitGen + tranPrep  + vm12 + 'XQ55000;' + dv0 + quiet + cleanup + '<XD;', ('Test for clean faults on shutoff.',)),
        'offSitHysBmsBB': (740, modEmptInitBB + slowTwitchDef + 'Xa-162;' + tranPrep + twitch + 'XQ568000;' + 'Pf;W2;Xa0;' + quiet + cleanup + '<XD;', ('for CompareRunRun.py Argon vs Photon builds. This is the only test for that.',)),
        'offSitHysBmsCHG': (800, modEmptInitCHG + slowTwitchDef + 'Xa-324;' + tranPrep + twitch + 'XQ568000;' + 'Pf;W2;Xa0;' + quiet + cleanup + '<XD;', ('for CompareRunRun.py Argon vs Photon builds. This is the only test for that.',)),
        'triTweakDisch': (262, slow + 'Rs;W4;Xp13;' + quiet + cleanup + '<XD;', ('Should run three very large current discharge/recharge cycles without fault', 'Best test for seeing time skews and checking fault logic for false trips', 'Occasional jumps in ib_sel_stat are normal when pass through 0 A.  Also hyst evident in one _s model')),
        'ampHiFailFf': (153, modHalfInit + tranPrep + 'Ff1;' + c50 + 'XQ40000;' + c00 + quiet + cleanup + '<XD;', ("Should detect but not switch amp current failure. (See 'diff' and current!=0 on display).", "Run about 60s. Start by looking at 'Ult 1'. No fault record (keeps recording).  Verify that on Fig 3 the e_wrap goes through a threshold ~0.4 without change of 'ib_sel_stat'", "This show when deploy with Fake Faults (Ff) don't throw false trips (it happened)", "ib_amp limited by max range e.g. 12.6.  ib_diff_fa will set red_loss but wait for wrap_fa to isolate and make selection change")),
        'ampLoFail': (150, modHalfInit + tranPrep + cm50 + 'XQ50000;' + c00 + quiet + cleanup + '<XD;', ("Should detect and switch amp current failure.", "Start looking at 'Ult 1'. Fault record (frozen). Will see 'diff' flashing on display even after fault cleared automatically (lost redundancy).", "ib_diff_fa will set red_loss but wait for wrap_fa to isolate and make selection change")),
        'ampLoFullFail': (153, modFullInit + tranPrep + cm50 + 'XQ50000;' + c00 + quiet + cleanup + '<XD;', ("Should detect and switch amp current failure before saturation tripped (would only be a problem for noa).", "Start looking at 'Ult 1'. Fault record (frozen). Will see 'diff' flashing on display even after fault cleared automatically (lost redundancy).", "ib_diff_fa will set red_loss but wait for wrap_fa to isolate and make selection change")),
        'noaLoFail': (153, modHalfInit + tranPrep + dm50 + 'XQ50000;' + c00 + quiet + cleanup + '<XD;', ("Should detect and switch amp current failure.", "Start looking at 'Ult 1'. Fault record (frozen). Will see 'diff' flashing on display even after fault cleared automatically (lost redundancy).", "ib_diff_fa will set red_loss but wait for wrap_fa to isolate and make selection change")),
        'noaLoFullFail': (153, modFullInit + 'DS-0.30' + tranPrep + dm50 + 'XQ50000;' + c00 + quiet + cleanup + '<XD;', ("Race with artificially low SAT logic to detect and switch amp current failure.", "Start looking at 'Ult 1'. Fault record (frozen). Will see 'diff' flashing on display.", "ib_diff_fa will set red_loss but wait for wrap_fa to isolate and make selection change")),
        'ampHiFailNoise': (107, modHalfInit + tranPrep + noisePackage + c50 + 'XQ25000;' + c00 + silentPackage + quiet + cleanup + '<XD;', ("Noisy ampHiFail.  Should detect and switch amp current failure.", "Start looking at 'Ult 1'. Fault record (frozen). Will see 'diff' flashing on display even after fault cleared automatically (lost redundancy).", "ib_diff_fa will set red_loss but wait for wrap_fa to isolate and make selection change")),
        'noaHiFailNoise': (107, modHalfInit + tranPrep + noisePackage + d50 + 'XQ25000;' + c00 + silentPackage + quiet + cleanup + '<XD;', ("Noisy ampHiFail.  Should detect and switch amp current failure.", "Start looking at 'Ult 1'. Fault record (frozen). Will see 'diff' flashing on display even after fault cleared automatically (lost redundancy).", "ib_diff_fa will set red_loss but wait for wrap_fa to isolate and make selection change")),
        'rapidTweakRegression40C': (262, 'D^15;' + slow + 'Rs;W4;Xp10;' + quiet + cleanup + '<XD;', ("Should run three very large current discharge/recharge cycles without fault", "Self-terminates", 'Occasional jumps in ib_sel_stat are normal when pass through 0 A')),
        'slowTweakRegression': (682, slow + 'Rs;W4;Xp11' + quiet + cleanup + '<XD;', ("Should run one very large slow (~15 min) current discharge/recharge cycle without fault.   It will take 60 seconds to start changing current.", 'Occasional jumps in ib_sel_stat are normal when pass through 0 A')),
        'satSitBB': (656, 'Xm247;Ca0.9962;' + fastTwitchDef + 'Xa17;' + tranPrep + 'XR;XQ600000;' + 'Xa0;' + quiet + cleanup + '<XD;', ("Should run one saturation and de-saturation event without fault.   Takes about 15 minutes.", "operate around saturation, starting below, go above, come back down. Tune Ca to start just below vsat",)),
        'satSitCHG': (656, 'Xm247;Ca0.986;' + fastTwitchDef + 'Xa17;' + tranPrep + 'XR;XQ600000;' + 'Xa0;' + quiet + cleanup + '<XD;', ("Should run one saturation and de-saturation event without fault.   Takes about 15 minutes.", "operate around saturation, starting below, go above, come back down. Tune Ca to start just below vsat",)),
        'flatSitHys': (680, 'Xm247;Ca0.9;Rb;Rf;Xts;Xa-81;Xf0.004;XW10000;XT10;XC2;W1;' + tranPrep + 'XR;XQ580000;Xa0;Xb0;' + quiet + cleanup + '<XD;', ("Operate around 0.9.  For CHINS, will check EKF with flat voc(soc).   Takes about 10 minutes.", "Make sure EKF soc (soc_ekf) tracks actual soc without wandering.")),
        'offSitHysBmsNoiseBB': (667, modEmptInitBB + slowTwitchDef + 'Xa-162;' + noisePackage + tranPrep + 'XR;XQ568000;' + 'Xa0;' + silentPackage + quiet + cleanup + '<XD;', ("Stress test with 2x normal Vb noise DV0.10.  Takes about 10 minutes.", "operate around saturation, starting above, go below, come back up. Tune Ca to start just above vsat. Go low enough to exercise hys reset ", "Make sure comes back on.", "It will show one shutoff only since becomes biased with pure sine input with half of down current ignored on first cycle during the shutoff.")),
        'offSitHysBmsNoiseCHG': (667, modEmptInitCHG + slowTwitchDef + 'Xa-324;' + noisePackage + tranPrep + 'XR;XQ568000;' + 'Xa0;' + silentPackage + quiet + cleanup + '<XD;', ("Stress test with 2x normal Vb noise DV0.10.  Takes about 10 minutes.", "operate around saturation, starting above, go below, come back up. Tune Ca to start just above vsat. Go low enough to exercise hys reset ", "Make sure comes back on.", "It will show one shutoff only since becomes biased with pure sine input with half of down current ignored on first cycle during the shutoff.")),
        # TODO for all volatile and saved parameters in Battery.csv:  'tranPrep' no 'vv' statment.  'stream' starts data incl 'vv'.  All adjusts before 'stream'
        'ampHiFailSlow': (515, modHalfInit + 'Fi3;Fc0.0006;Fd0.5;' + tranPrep + c10 + 'XQ400000;' + c00 + quiet + cleanup + '<XD;', ("10A bias on amp, disable wrap, noa in range at 0A and reflects battery state. Artificially tight cc_diff threshold.  Will detect diff but no wrap. Will be slow (~6 min) cc_diff detection as it waits for the EKF to wind up to produce a cc_diff fault and complete isolation and switch to noa.", "EKF should tend to follow voltage while soc wanders away.", "Run for 6  minutes to see that cc_diff_fa does set")),
        'noaHiFailSlow': (515, modHalfInit+ 'Fc0.0006;' + tranPrep + d20 + 'XQ400000;' + c00 + quiet + cleanup + '<XD;', ("20A bias on noa, amp in range at 0A and reflects battery state. Artificially tight cc_diff threshold. Will detect and switch noa current failure due to wrap+diff. Once wrap trips diff won't be displayed. Cannnot ever produce a cc_diff fault because amp still used.", "Will display “diff” due to 20A difference.", "EKF won't move because fed by amp.", "Run for 6  minutes to verify not cc_diff_fa")),
        'noaHiFailSlower': (515, modHalfInit+ 'Fc0.0006;' + tranPrep + d08 + 'XQ400000;' + c00 + quiet + cleanup + '<XD;', ("8A bias on noa, amp in range at 0A and reflects battery state.  Artificially tight cc_diff threshold. Will detect and switch noa current failure due to wrap+diff. Once wrap trips diff won't be displayed. Cannnot ever produce a cc_diff fault.", "Will display “diff” due to 6 A difference..", "EKF won't move because fed by amp.", "Run for 6  minutes to see potential cc_diff_fa")),
        'noaHiFailSlowest': (515, modHalfInit+ 'Fc0.0006;' + tranPrep + d05 + 'XQ400000;' + c00 + quiet + cleanup + '<XD;', ("5A bias on noa, amp in range at 0A and reflects battery state. Artificially tight cc_diff threshold. Not enough current to trip the noa wrap.  Cannnot ever produce a cc_diff fault because amp still used.", "Will display “diff” due to 5 A difference..", "EKF won't move because fed by amp.", "Run for 6  minutes to see potential cc_diff_fa")),
        'vHiFail': (138, modHalfInit + tranPrep + 'XY;Dv0.82;XQ60000;' + dv0 + quiet + cleanup + '<XD;', ("Should detect voltage failure and display '*fail' and 'redl' within 60 seconds.", "To diagnose, begin with 'Ult 1'.   Look for e_wrap to go through ewlo_thr.", "You may have to increase magnitude of injection (Dv).  The threshold is 32 * r_ss.", "There MUST be no SATURATION")),
        'vHiFailNoise': (138, modHalfInit + noisePackage + tranPrep + 'XY;Dv0.82;XQ60000;' + dv0 + quiet + cleanup + '<XD;', ("Should detect voltage failure and display '*fail' and 'redl' within 60 seconds.", "To diagnose, begin with 'Ult 1'.   Look for e_wrap to go through ewlo_thr.", "You may have to increase magnitude of injection (Dv).  The threshold is 32 * r_ss.", "There MUST be no SATURATION")),
        'vHiFailH': (84, modHalfInit + tranPrep + 'SH.3;W10;' + 'XY;Dv0.82;XQ30000;' + dv0 + quiet + cleanup + '<XD;', ("Should detect voltage failure and display '*fail' and 'redl' within 60 seconds.", "To diagnose, begin with 'Ult 1'.   Look for e_wrap to go through ewlo_thr.", "You may have to increase magnitude of injection (Dv).  The threshold is 32 * r_ss.", "There MUST be no SATURATION.  Initial BB shift will be limited by hys table")),
        'vHiFailFf': (138, modHalfInit + tranPrep + 'Ff1;XY;Dv0.8;XQ60000;' + dv0 + quiet + cleanup + '<XD;', ("Run for about 1 minute.", "Should detect voltage failure (see DOM1) but not display anything on display.", "Usually shows SAT.")),
        'pulseSSH': (25, synced_slow + 'Xp8;' + quiet + cleanup + '<XD;', ("Should generate a very short <10 sec data burst with a hw pulse.  Look at plots for good overlay. e_wrap should be flat.", "This is the shortest of all tests.  Useful for quick checks.", "ib_diff_flt will take time beyond event to reset running Hi-Lo.")),
        'tLoFailHdwe': (300, modHalfInit + 'Xm230;' + tranPrep + 'XY;Dt-113;XQ120000;' + 'Dt0;Rf;W50;' + cleanup + '<W50;' + quietwait + '<Pf;<XD;', ("Simulates open thermistor.", "To diagnose, begin with 'Ult 1'.   Look for e_wrap to go through ewlo_thr.", "You may have to increase magnitude of injection (Dv).  The threshold is 32 * r_ss.", "There MUST be no SATURATION")),
        'DvMon': (152, modHalfInit + tranPrep + 'XY;Dw-0.8;Dn0.0001;XQ120000;Dw0;Rf;W50;' + quiet + cleanup + '<XD;', ("Should detect and switch voltage failure and use vb_model", "'*fail' will be displayed.", "To evaluate plots, start looking at 'Ult 1'. Fault record (frozen). Will see 'redl' flashing on display even after fault cleared automatically (lost redundancy).", "Run for 2 min to confirm no cc_diff_fa")),
        'DvSim': (152, modHalfInit + tranPrep + 'XY;Dy-0.8;Dn0.0001;XQ120000;Dy0;Rf;W50;' + quiet + cleanup + '<XD;', ("Should detect and switch voltage failure and use vb_model", "'*fail' will be displayed.", "To evaluate plots, start looking at 'Ult 1'. Fault record (frozen). Will see 'redl' flashing on display even after fault cleared automatically (lost redundancy).", "Run for 2 min to confirm no cc_diff_fa")),
        'faultParade': (320, modHalfInit + 'Dh1000;vv4;W4;XY;Dm50;Dn0.0001;W200;Dm0;Dn0;W20;Rf;XQ240000;' + quiet + cleanup + '<XD;', ("Check fault, history, and summary logging", "Should flag faults but take no action", "", "", "")),
        'stepDown': (103, modHalfInit + tranPrep + sd50 + 'XQ25000;' + s00 + quiet + cleanup + '<XD;', ("Should be normal hard discharge step", "", "", "")),
        'stepUp': (103, modHalfInit + tranPrep + sc50 + 'XQ25000;' + s00 + quiet + cleanup + '<XD;', ("Should be normal hard charge step", "", "", "")),
        'zero_with_pc': (113, hdwNoVbPcMidInit + zeroPrepHdweNoVb + 'vv4;W17;' + 'XQ25000;' + 'vv99;Xm2;XQ15000;' + quiet  + cleanup + '<XD;', ("Hardware zero_with_pc run", "", "", "")),
        }

macro_lookup = {
        'end_early': (22, 'Y;cc;Dh1800000;*W;*vv0;*XS;*Ca1;<Hd;<Pf;', ('', '', '', '')),
        'hdwNoVbPcMidInit': (5, hdwNoVbPcMidInit, ('', '', '', '')),
        'modFullInit': (5, modFullInit, ('', '', '', '')),
        'modLoInit': (5, modLoInit, ('', '', '', '')),
        'modHalfInit': (5, modHalfInit, ('', '', '', '')),
        'modEmptInitBB': (5, modEmptInitBB, ('', '', '', '')),
        'noisePackage': (5, noisePackage, ('', '', '', '')),
        'silentPackage': (5, silentPackage, ('', '', '', '')),
        'quiet': (5, quiet, ('', '', '', '')),
        'cleanup': (5, cleanup, ('', '', '', '')),
        'tempCleanup': (5, tempCleanup, ('', '', '', '')),
        'tranPrep': (5, tranPrep, ('', '', '', '')),
        'zeroPrepHdweNoVb': (5, zeroPrepHdweNoVb, ('', '', '', '')),
        'zero_set_hdwe_no_Vb': (5, zero_set_hdwe_no_Vb, ('', '', '', '')),
        'time_stamp': (5, time_stamp, ('', '', '', '')),
        'synced_slow': (5, synced_slow, ('', '', '', '')),
        'slowTwitchDef': (5, slowTwitchDef, ('', '', '', '')),
        'fastTwitchDef': (5, fastTwitchDef, ('', '', '', '')),
        'c06': (5, c06, ('', '', '', '')),
        'd06': (5, d06, ('', '', '', '')),
        'c08': (5, c08, ('', '', '', '')),
        'd05': (5, d05, ('', '', '', '')),
        'd08': (5, d08, ('', '', '', '')),
        'c10': (5, c10, ('', '', '', '')),
        'd10': (5, d10, ('', '', '', '')),
        'c18': (5, c18, ('', '', '', '')),
        'd18': (5, d18, ('', '', '', '')),
        'c50': (5, c50, ('', '', '', '')),
        'd50': (5, d50, ('', '', '', '')),
        'cm50': (5, cm50, ('', '', '', '')),
        'c00': (5, c00, ('', '', '', '')),
        'dv0': (5, dv0, ('', '', '', '')),
        'twitch': (5, twitch, ('', '', '', '')),
        'noaHiFail': (5, d50, ('', '', '', '')),
        'noaHiFailNoise': (5, d50, ('', '', '', '')),
        }

plink_connection = {'': 'test',
                    'soc0p': 'testsoc0p',
                    'soc1a': 'testsoc1a',
                    'pro0p': 'testpro0p',
                    'pro1a': 'testpro1a',
                    'pro2p2': 'testpro2p2',
                    'pro2p2_hi_lo': 'testpro2p2',
                    'pro3p2': 'testpro3p2',
                    'pro3p2_hi_lo': 'testpro3p2',
                    'pro4p2': 'testpro4p2',
                    'soc2p2': 'testsoc2p2',
                    'soc2p2_hi_lo': 'testsoc2p2',
                    'soc3p2': 'testsoc3p2',
                    'soc3p2_hi_lo': 'testsoc3p2',
                    'soc4p2': 'testsoc4p2',
                    'soc4p2_hi_lo': 'testsoc4p2',
                    }


# Begini - configuration class using .ini files
class Begini(ConfigParser):

    def __init__(self, name, def_dict_):
        ConfigParser.__init__(self)

        config_path, config_basename = str(PurePosixPath(name).parent), PurePosixPath(name).name
        if platform.system() == 'Linux':
            config_txt = PurePosixPath(config_basename).stem + '_linux.ini'
            self.config_file_path = str(PurePosixPath('/home/daveg/.local') / config_txt)
        elif platform.system() == 'Darwin':
            config_txt = PurePosixPath(config_basename).stem + '_macos.ini'
            self.config_file_path = str(PurePosixPath('/Users/daveg/.local') / config_txt)
        else:
            config_txt = PurePosixPath(config_basename).stem + '.ini'
            local_app_data = os.getenv('LOCALAPPDATA') or str(Path.home() / 'AppData' / 'Local')
            self.config_file_path = str(Path(local_app_data) / config_txt)
        print('config file', self.config_file_path)
        if Path(self.config_file_path).is_file():
            self.read(self.config_file_path)
        else:
            with open(self.config_file_path, 'w') as cfg_file:
                self.read_dict(def_dict_)
                self.write(cfg_file)
            print('wrote', self.config_file_path)

    # Get an item
    def get_item(self, ind, item):
        return self[ind][item]

    # Put an item
    def put_item(self, ind, item, value):
        self[ind][item] = value
        self.save_to_file()

    # Save again
    def save_to_file(self):
        with open(self.config_file_path, 'w') as cfg_file:
            self.write(cfg_file)
        print('wrote', self.config_file_path)


# Executive class to control the global variables
class ExRoot:
    def __init__(self):
        self.script_loc = Path(__file__).resolve().parent.as_posix()
        self.config_path = str(PurePosixPath(self.script_loc) / 'root_config.ini')
        self.version = None
        self.root_config = None
        self.load_root_config(self.config_path)

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


# Global methods
def add_to_clip_board(text):
    pyperclip.copy(text)


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


# Split all information contained in file path
def contain_all(testpath):
    folder_path, basename = str(PurePosixPath(testpath).parent), PurePosixPath(testpath).name
    parent, txt = str(PurePosixPath(folder_path).parent), PurePosixPath(folder_path).name
    # get key
    key = ''
    with open(testpath, 'r') as file:
        for line in file:
            if line.__contains__(txt):
                shorter = line[line.find(txt):]
                end_key = shorter.find(',')
                key = shorter[:end_key].strip()
                break
    return folder_path, parent, basename, txt, key


# plink generates '\0' characters
def copy_clean(src, dst):
    with open(src, 'r') as file_in:
        data = file_in.read()
    with open(dst, 'w') as file_out:
        file_out.write(data.replace('\0', ''))


def create_file_key(version_, unit_, battery_):
    return version_ + '_' + unit_ + '_' + battery_


def create_file_txt(option_, unit_, battery_):
    return option_ + '_' + unit_ + '_' + battery_ + '.csv'


def empty_file(target):
    # create empty file
    try:
        with open(target, 'w') as _:
            pass
    except Exception as e:
        print(f"empty_file: failed to empty {target} with {e}")
    print('emptied', target)


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
    start_button.config(bg=bg_color, activebackground=bg_color, fg='black', activeforeground='purple')
    get_time_button.config(bg=bg_color, activebackground=bg_color, fg='black', activeforeground='purple')


def grab_init(command_to_append='', force_if_ready=False):
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
    return start_plink(command_to_paste=init_command, force_if_ready=force_if_ready)


def monitor_plink_done():
    if Path(plink_test_csv_path.get()).is_file():
        try:
            with open(plink_test_csv_path.get(), 'rb') as f:
                f.seek(0, 2)
                size = f.tell()
                # Read last 1024 bytes to check for ***DONE***
                f.seek(max(0, size - 1024))
                last_data = f.read().decode('utf-8', errors='ignore')
                if '***DONE***' in last_data:
                    print(f"***DONE*** detected in {plink_test_csv_path.get()}")
                    save_data()
                    tk.messagebox.showinfo(title='Done ' + start_button.cget('text'), message='Run Complete')
                    return
        except Exception as e:
            print(f"Error monitoring plink file: {e}")
    master.after(1000, monitor_plink_done)


def grab_start():
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
        if not grab_init(command_to_append=start_command, force_if_ready=True):
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
    monitor_plink_done()


def grab_all_nominal():
    macro_button.config(bg=bg_color, activebackground=bg_color, fg='black', activeforeground='black')
    init_button.config(bg=bg_color, activebackground=bg_color, fg='black', activeforeground='purple')
    start_button.config(bg=bg_color, activebackground=bg_color, fg='black', activeforeground='purple')
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
    start_button.config(bg=bg_color, activebackground=bg_color, fg='black', activeforeground='purple')
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


def kill_plink(sys_=None, silent=True):
    global plink_pid
    command = ''
    if plink_pid:
        if sys_ == 'Windows':
            command = f'taskkill /f /pid {plink_pid} /t'
            print(f"Terminating Plink with command: {command}")
            try:
                run_shell_cmd(command, silent=silent)
                plink_pid = None
                return 0
            except Exception as e:
                print(f"Error killing PID {plink_pid}: {e}")
                plink_pid = None
        else:
            # On Linux/macOS, find the parent PID (PPID)
            ppid = None
            try:
                # Get the parent PID using ps
                ppid_out = subprocess.check_output(['ps', '-o', 'ppid=', '-p', str(plink_pid)]).decode().strip()
                if ppid_out:
                    ppid = int(ppid_out)
            except Exception:
                pass

            if ppid and ppid > 1: # Avoid killing init (PID 1)
                # Kill both the process and its parent
                command = f'kill -9 {plink_pid} {ppid}'
                print(f"Terminating Plink and parent terminal with command: {command}")
            else:
                command = f'kill -9 {plink_pid}'
                print(f"Terminating Plink with command: {command}")

            try:
                run_shell_cmd(command, silent=silent)
                plink_pid = None
                return 0
            except Exception as e:
                print(f"Error killing PID {plink_pid}: {e}")
                plink_pid = None

    # If we reached here, either plink_pid was None or we want to be sure
    command = ''
    if sys_ == 'Linux':
        command = 'pkill -e plink; pkill -f "gnome-terminal --zoom=0.8"'
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
        with open(auto_plink_path, 'w') as f:
            f.write('#folder, version, unit, battery, macro,\n')
        print(f"Prepopulated {auto_plink_path} with header.")
    print(f"Report: plink_test.csv location is {plink_path}")


def plink_size():
    if Path(plink_test_csv_path.get()).is_file():
        enter_size = Path(plink_test_csv_path.get()).stat().st_size  # bytes
    else:
        enter_size = 0
    return enter_size


def grab_auto():
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
                header_line = line
                header_fields = [f.strip() for f in line.lstrip('#').split(',')]
                # Filter out empty fields that might occur if there's a trailing comma
                header_fields = [f for f in header_fields if f]
                print(f"Fields: {header_fields}")
                continue
            
            if header_line is not None:
                # Validation: number of fields separated by commas
                # header line example: "#folder, version, unit, battery, macro,"
                # We expect the same number of comma-separated fields in each data line.
                header_field_count = len(header_line.split(','))
                line_field_count = len(line.split(','))
                
                if line_field_count != header_field_count:
                    print(f"Skipping line due to field count mismatch ({line_field_count} vs {header_field_count}): {line}")
                    continue
                
                values = [v.strip() for v in line.split(',')]
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

        msg_label = tk.Label(dialog, text=all_lines_text, justify='left', font=('Courier', 10))
        msg_label.pack(padx=20, pady=20, fill='both', expand=True)

        prompt_label = tk.Label(dialog, text="Do you want to run these configurations automatically?", font=('Arial', 11, 'bold'))
        prompt_label.pack(pady=10)

        result = tk.BooleanVar(value=False)

        def on_yes():
            result.set(True)
            dialog.destroy()

        def on_no():
            result.set(False)
            dialog.destroy()

        btn_frame = tk.Frame(dialog)
        btn_frame.pack(pady=20)
        tk.Button(btn_frame, text="Yes", width=10, command=on_yes).pack(side='left', padx=10)
        tk.Button(btn_frame, text="No", width=10, command=on_no).pack(side='left', padx=10)

        master.wait_window(dialog)
        
        if not result.get():
            print("User response for automatic run: No")
            return
        
        print("User response for automatic run: Yes")
        
        # Save configuration
        saved_config = {
            'folder': Test.dataReduction_folder,
            'version': Test.version,
            'unit': test_unit.get(),
            'battery': test_battery.get(),
            'macro': macro_option.get(),
            'option': option.get()
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

        # Process each line
        for values in data_rows:
            # Map values to fields
            config = {}
            for i in range(min(len(header_fields), len(values))):
                config[header_fields[i]] = values[i]

            print(f"Processing configuration: {config}")

            # Mapping of header fields to GUI actions
            # Expected headers: folder, version, unit, battery, macro
            
            if 'folder' in config:
                Test.dataReduction_folder = config['folder']
                Test.update_folder_button()
                set_red(Test.folder_button)
                tkinter.messagebox.showinfo("Update", f"Changed folder to: {config['folder']}")

            if 'version' in config:
                Test.version = config['version']
                Test.update_version_button()
                set_red(Test.version_button)
                tkinter.messagebox.showinfo("Update", f"Changed version to: {config['version']}")

            if 'unit' in config:
                test_unit.set(config['unit'])
                set_red(Test.unit_button)
                tkinter.messagebox.showinfo("Update", f"Changed unit to: {config['unit']}")

            if 'battery' in config:
                test_battery.set(config['battery'])
                set_red(Test.battery_button)
                tkinter.messagebox.showinfo("Update", f"Changed battery to: {config['battery']}")

            if 'macro' in config:
                if config['macro'] in macro_lookup:
                    macro_option.set(config['macro'])
                    set_red(macro_sel)
                    tkinter.messagebox.showinfo("Update", f"Changed macro to: {config['macro']}")
                elif config['macro'] in lookup:
                    # If it's in 'lookup' but not 'macro_lookup', it's likely intended as an 'option'
                    option.set(config['macro'])
                    set_red(sel)
                    set_red(sel1)
                    lookup_start()
                    tkinter.messagebox.showinfo("Update", f"Changed option to: {config['macro']}")
                else:
                    print(f"Error: Macro '{config['macro']}' not found in macro_lookup or lookup. Skipping.")
                    tkinter.messagebox.showerror("Error", f"Macro '{config['macro']}' is invalid and will be skipped.")

        # Restore configuration
        print(f"Restoring configuration: {saved_config}")
        Test.dataReduction_folder = saved_config['folder']
        Test.update_folder_button()
        Test.version = saved_config['version']
        Test.update_version_button()
        test_unit.set(saved_config['unit'])
        test_battery.set(saved_config['battery'])
        macro_option.set(saved_config['macro'])
        option.set(saved_config['option'])
        
        # Restore colors
        Test.folder_button.config(fg='blue', activeforeground='blue')
        Test.version_button.config(fg='blue', activeforeground='blue')
        Test.unit_button.config(fg='black', activeforeground='black')
        Test.battery_button.config(fg='black', activeforeground='black')
        macro_sel.config(fg='black', activeforeground='black')
        sel.config(fg='black', activeforeground='black')
        sel1.config(fg='black', activeforeground='black')

        lookup_start()
        tkinter.messagebox.showinfo("Restore", "Restored previous configuration.")

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


def save_data():
    global timer
    print(f"save_data: {plink_test_csv_path.get()=}")
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
        if auto_overwrite.get():
            print('auto over-write triggering comparison')
            compare_run()
    else:
        print('plink test file non-existent or too small (<64 bytes) probably already done')
        tkinter.messagebox.showwarning(message="Nothing to save")
    start_button.config(bg=bg_color, activebackground=bg_color, fg='black', activeforeground='purple')


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
    start_button.config(bg=bg_color, activebackground=bg_color, fg='black', activeforeground='purple')


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


def size_of(path):
    if Path(path).is_file() and (size := Path(path).stat().st_size) > 0:  # bytes
        return size
    else:
        return 0


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
                if '***READY***\n' in last_data:
                    return True
        except Exception as e:
            print(f"Error checking plink ready: {e}")
    return False


def start_plink(command_to_paste=None, force_if_ready=False):
    global plink_pid
    lookup_test()
    if look_plink(platform.system()):
        if force_if_ready:
            if is_plink_ready():
                print("Plink is READY. Restarting to automate command.")
                kill_plink(platform.system())
                tksleep(0.5)  # Give some time for the process to exit
                # Proceed to restart logic below
            else:
                print("Plink is already open but NOT READY.")
                tkinter.messagebox.showinfo(title="Not Ready",
                                           message="Please wait until terminal is READY or run the START HERE button.")
                return False
        else:
            print("Plink already open. Skipping restart.")
            if command_to_paste:
                print(f"Plink already open. CANNOT automatically paste command: {command_to_paste}")
                print("Please paste it manually into the terminal.")
            return True

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
            plink_base_cmd = f"plink -load {test_filename.get()} | tee {plink_test_csv_path.get()}"
            if command_to_paste:
                # (echo 'command'; cat) | plink ... ensures the command is sent and the session remains interactive
                plink_cmd = f"(echo '{command_to_paste}'; cat) | {plink_base_cmd}; exec bash"
            else:
                plink_cmd = f"{plink_base_cmd}; exec bash"

            if 'gnome-terminal' in term:
                # zoom 0.8 is roughly "two sizes smaller" (standard is 1.0, 0.9 is one size, 0.8 is two)
                cmd = [term, '--zoom=0.8', '--', 'bash', '-c',
                       f"echo -e '\\e]11;#000000\\a\\e]10;#00ff00\\a'; clear; {plink_cmd}"]
                print(f"Running command: {shlex.join(cmd)}")
                proc = subprocess.Popen(cmd)
                tksleep(1.0) # Wait for terminal to spawn plink
                try:
                    # Debug: Print full result of pgrep and ps -ef | grep plink
                    pgrep_search = f"plink -load {test_filename.get()}"
                    print(f"Debug: pgrep -a -f result for '{pgrep_search}':")
                    try:
                        pgrep_out = subprocess.check_output(['pgrep', '-a', '-f', pgrep_search]).decode()
                        print(pgrep_out)
                    except subprocess.CalledProcessError:
                        print("No processes found with pgrep.")

                    print(f"Debug: ps -ef | grep plink result:")
                    try:
                        ps_out = subprocess.check_output(['ps', '-ef']).decode()
                        # Filter lines containing 'plink' excluding the grep/ps process if possible
                        plink_lines = [line for line in ps_out.splitlines() if 'plink' in line and 'grep' not in line]
                        print("\n".join(plink_lines))
                    except Exception as e:
                        print(f"Error running ps: {e}")

                    # Find the newest plink process matching our session
                    out = subprocess.check_output(['pgrep', '-n', '-f', pgrep_search]).decode().strip()
                    if out:
                        plink_pid = int(out)
                except Exception:
                    plink_pid = proc.pid
                
                # Get the parent PID using ps
                ppid = "Unknown"
                try:
                    ppid_out = subprocess.check_output(['ps', '-o', 'ppid=', '-p', str(plink_pid)]).decode().strip()
                    if ppid_out:
                        ppid = ppid_out
                except Exception:
                    pass
                print(f"Spawned PID: {plink_pid}  PPID: {ppid}")
            elif 'xterm' in term:
                # xterm -bg black -fg green -fs 10 (assuming default is ~12)
                cmd = [term, '-bg', 'black', '-fg', 'green', '-fs', '10', '-e', f"bash -c '{plink_cmd}'"]
                print(f"Running command: {shlex.join(cmd)}")
                proc = subprocess.Popen(cmd)
                tksleep(1.0) # Wait for terminal to spawn plink
                try:
                    # Debug: Print full result of pgrep and ps -ef | grep plink
                    pgrep_search = f"plink -load {test_filename.get()}"
                    print(f"Debug: pgrep -a -f result for '{pgrep_search}':")
                    try:
                        pgrep_out = subprocess.check_output(['pgrep', '-a', '-f', pgrep_search]).decode()
                        print(pgrep_out)
                    except subprocess.CalledProcessError:
                        print("No processes found with pgrep.")

                    print(f"Debug: ps -ef | grep plink result:")
                    try:
                        ps_out = subprocess.check_output(['ps', '-ef']).decode()
                        plink_lines = [line for line in ps_out.splitlines() if 'plink' in line and 'grep' not in line]
                        print("\n".join(plink_lines))
                    except Exception as e:
                        print(f"Error running ps: {e}")

                    out = subprocess.check_output(['pgrep', '-n', '-f', pgrep_search]).decode().strip()
                    if out:
                        plink_pid = int(out)
                except Exception:
                    plink_pid = proc.pid
                
                # Get the parent PID using ps
                ppid = "Unknown"
                try:
                    ppid_out = subprocess.check_output(['ps', '-o', 'ppid=', '-p', str(plink_pid)]).decode().strip()
                    if ppid_out:
                        ppid = ppid_out
                except Exception:
                    pass
                print(f"Spawned PID: {plink_pid}  PPID: {ppid}")
            else:
                cmd = [term, '-e', f"bash -c '{plink_cmd}'"]
                print(f"Running command: {shlex.join(cmd)}")
                proc = subprocess.Popen(cmd)
                tksleep(1.0) # Wait for terminal to spawn plink
                try:
                    # Debug: Print full result of pgrep and ps -ef | grep plink
                    pgrep_search = f"plink -load {test_filename.get()}"
                    print(f"Debug: pgrep -a -f result for '{pgrep_search}':")
                    try:
                        pgrep_out = subprocess.check_output(['pgrep', '-a', '-f', pgrep_search]).decode()
                        print(pgrep_out)
                    except subprocess.CalledProcessError:
                        print("No processes found with pgrep.")

                    print(f"Debug: ps -ef | grep plink result:")
                    try:
                        ps_out = subprocess.check_output(['ps', '-ef']).decode()
                        plink_lines = [line for line in ps_out.splitlines() if 'plink' in line and 'grep' not in line]
                        print("\n".join(plink_lines))
                    except Exception as e:
                        print(f"Error running ps: {e}")

                    out = subprocess.check_output(['pgrep', '-n', '-f', pgrep_search]).decode().strip()
                    if out:
                        plink_pid = int(out)
                except Exception:
                    plink_pid = proc.pid
                
                # Get the parent PID using ps
                ppid = "Unknown"
                try:
                    ppid_out = subprocess.check_output(['ps', '-o', 'ppid=', '-p', str(plink_pid)]).decode().strip()
                    if ppid_out:
                        ppid = ppid_out
                except Exception:
                    pass
                print(f"Spawned PID: {plink_pid}  PPID: {ppid}")
        elif platform.system() == 'Windows':
            # 'color 0A' sets black background (0) and light green foreground (A)
            plink_base_cmd = f"plink -load {test_filename.get()} -tee {plink_test_csv_path.get()}"
            if command_to_paste:
                # (echo command & type CON) | plink ... is the Windows equivalent for sending a command then remaining interactive
                plink_cmd = f"(echo {command_to_paste} & type CON) | {plink_base_cmd}"
            else:
                plink_cmd = plink_base_cmd

            cmd = ['cmd', '/c', 'start', 'cmd', '/k', f"color 0A && {plink_cmd}"]
            print(f"Running command: {' '.join(cmd)}")
            proc = subprocess.Popen(cmd)
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
                plink_pid = proc.pid
            
            # Get the parent PID using ps
            ppid = "Unknown"
            try:
                # tasklist doesn't easily give PPID without additional tools or WMI, 
                # but we can try using 'wmic process where processid=... get parentprocessid'
                wmic_out = subprocess.check_output(['wmic', 'process', 'where', f'processid={plink_pid}', 'get', 'parentprocessid']).decode('ascii')
                lines = wmic_out.strip().split('\n')
                if len(lines) > 1:
                    ppid = lines[1].strip()
            except Exception:
                pass
            print(f"Spawned PID: {plink_pid}  PPID: {ppid}")
        elif platform.system() == 'Darwin':
             if command_to_paste:
                 plink_cmd = f"(echo '{command_to_paste}'; cat) | plink -load {test_filename.get()} -tee {plink_test_csv_path.get()}"
             else:
                 plink_cmd = f"plink -load {test_filename.get()} -tee {plink_test_csv_path.get()}"

             script = (f'tell application "Terminal" to do script '
                       f'"printf \\"\\\\e]11;#000000\\\\a\\\\e]10;#00ff00\\\\a\\"; clear; '
                       f'{plink_cmd}"\n'
                       f'tell application "Terminal" to set font size of window 1 to 10')
             cmd = ['osascript', '-e', script]
             print(f"Running command: {shlex.join(cmd)}")
             proc = subprocess.Popen(cmd)
             tksleep(1.0)
             try:
                 out = subprocess.check_output(['pgrep', '-n', '-f', f"plink -load {test_filename.get()}"]).decode().strip()
                 if out:
                     plink_pid = int(out)
             except Exception:
                 plink_pid = proc.pid
             
             # Get the parent PID using ps
             ppid = "Unknown"
             try:
                 ppid_out = subprocess.check_output(['ps', '-o', 'ppid=', '-p', str(plink_pid)]).decode().strip()
                 if ppid_out:
                     ppid = ppid_out
             except Exception:
                 pass
             print(f"Spawned PID: {plink_pid}  PPID: {ppid}")
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
    start_button.config(bg=bg_color, activebackground=bg_color, fg='black', activeforeground='purple')


if __name__ == '__main__':  # Example usage.  Ran ok 20260217
    import os
    import tkinter as tk
    from tkinter import ttk

    ex_root = ExRoot()

    cf = Begini(__file__, def_dict)

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
        init_button = myButton(option_panel_ctr, text='START HERE and PASTE then\n wait for temp init complete', command=grab_init, fg="purple", bg=bg_color,
                               justify='left', font=("Arial", 8))
    else:
        init_button = myButton(option_panel_ctr, text='START HERE and PASTE then\n wait for temp init complete', command=grab_init, fg="purple", bg=bg_color,
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
        start_button = myButton(option_panel_ctr, text='', command=grab_start, fg="purple", bg=bg_color,
                                justify='left', font=butt_font)
        prev_button = myButton(option_panel_right, text='Run Prev', command=run_previous_task, fg="blue", bg=bg_color,
                                justify='left', font=butt_font)
    else:
        start_button = myButton(option_panel_ctr, text='', command=grab_start, fg="purple", bg=bg_color, wraplength=wrap_length,
                                justify='left', font=butt_font)
        prev_button = myButton(option_panel_right, text='Run Prev', command=run_previous_task, fg="blue", bg=bg_color, wraplength=wrap_length,
                                justify='left', font=butt_font)
    start_button.pack(padx=5, pady=5, expand=True, fill='both')
    prev_button.pack(side='left', padx=5, pady=5)
    auto_button = myButton(option_panel_right, text='AUTO', command=grab_auto, fg="blue", bg=bg_color,
                           justify='left', font=butt_font)
    auto_button.pack(side='left', padx=5, pady=5)
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
