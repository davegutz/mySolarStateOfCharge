# GUI_common - shared imports for both GUI drivers
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
import os
import platform
import sys
from configparser import ConfigParser
from pathlib import Path, PurePosixPath
import pyperclip

plat = sys.platform

if plat == 'linux':
    default_dr = '/home/daveg/gdrive/GitHubArchive/SOC_Particle/dataReduction'
elif plat == 'darwin':
    default_dr = '/Users/daveg/Library/CloudStorage/GoogleDrive-davegutz2006@gmail.com/My Drive/GitHubArchive/SOC_Particle/dataReduction'
else:
    default_dr = 'G:/My Drive/GitHubArchive/SOC_Particle/dataReduction'


# Configuration for entire folder selection read with filepaths
default_dict = {
    'test': {
        "version": "g2025012a",
        "unit": "soc3p2",
        "battery": "bb",
        'dataReduction_folder': default_dr,
    },
    'ref': {
        "version": "g2025012a",
        "unit": "soc3p2",
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
        'hardcopy': False,
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
    'custom', 'zero_with_pc', 'ampHiEmptFail', 'ampHiFail', 'noaHiFail', 'rapidTweakRegression',
    'pulseSoft', 'pulseHard', 'rapidTweakRegressionH0', 'offLowSoc', 'offSitBmsBB',
    'offSitBmsCHG', 'triTweakDisch', 'ampHiFailFf', 'ampLoFail', 'ampLoFullFail', 'noaLoFail', 'noaLoFullFail',
    'ampHiFailNoise', 'noaHiFailNoise',
    'rapidTweakRegression40C', 'slowTweakRegression', 'satSitBB', 'satSitCHG',
    ]
sel_list1 = [
    'flatSit', 'offSitBmsNoiseBB', 'offSitBmsNoiseCHG', 'ampHiFailSlow',
    'noaHiFailSlow', 'noaHiFailSlower', 'noaHiFailSlowest', 'vHiFail', 'vHiFailNoise', 'vHiFailFf',
    'pulseHard', 'tLoFailModel', 'tHiFailModel', 'tLoFailHdwe', 'tHiFailHdwe', 'faultParade', 'stepDown', 'stepUp',
    'ibDualMid', 'ibDualFlat', 'vcFlat',
    ]

# Default content for auto_plink.csv (analogous to default_dict for the .ini file)
default_auto_header = 'folder, version, battery, macro, hardfigure'
_auto_row = {'folder': default_dr, 'version': 'g20250612a', 'battery': 'bb', 'hardfigure': 'True'}
default_auto = (
    [{**_auto_row, 'macro': 'ampHiEmptFail'},
     {**_auto_row, 'macro': 'ampHiFail'},
     {**_auto_row, 'macro': 'noaHiFail'}] +
    [{**_auto_row, 'macro': m} for m in sel_list[sel_list.index('rapidTweakRegression'):]] +
    [{**_auto_row, 'macro': m} for m in sel_list1[:sel_list1.index('vcFlat') + 1]]
)

macro_sel_list = [
    'end_early', 'hdwNoVbPcMidInit', 'modHalfInit', 'modHalfInit230', 'modHalfInit239', 'modEmptInitBB', 'modEmptInitCHG',
    'noisePackage', 'silentPackage', 'quiet', 'cleanup', 'tempCleanup', 'tranPrep', 'synced_slow', 'slow',
    'slowTwitchDef', 'fastTwitchDef', 'c06', 'd06', 'c08', 'd05', 'd08', 'c10', 'd10', 'c18', 'd18', 'c50', 'cm50',
    'cmn50', 'dmn50', 'cmn100', 'dmn100',
    'c00', 'dv0', 'twitch', 'time_stamp', 's00', 'sd50', 'sc50', 'zeroPrepHdweNoVb', 'zero_set_hdwe_no_Vb',
    'noaHiFail', 'noaHiFailNoise',
    ]

# Macro
satInit = 'Dh;*W;*vv0;*XS;*Ca1;BZ;Ff0;DP1;HR;Rf;'
hdwNoVbPcMidInit = 'vv0;Xm2;Ca0.50;BZ;Ff0;W20;DP1;HR;Rf;'
modFlatInit = 'vv0;Xm247;Ca0.90;BZ;Ff0;DP1;HR;Rf;'
modFlatInitHi = 'vv0;Xm3;Ca0.90;BZ;Ff0;DP1;HR;Rf;'
modFullInit = 'vv0;Xm247;Ca0.93;BZ;Ff0;DP1;HR;Rf;'  # kickers off 0.94
modLoInit = 'vv0;Xm247;Ca0.17;BZ;Ff0;DP1;HR;Rf;'
modHalfInit = 'vv0;Xm247;Ca0.50;BZ;Ff0;DP1;HR;Rf;'
modHalfInit230 = 'Pv;Pr;-vv-1;-Xm230;Pm;Ps;-Ca0.50;Pm;Ps;BZ;Ff0;DP1;HR;Rf;'
modHalfInit239 = '-vv0;-Xm239;-Ca0.50;BZ;Ff0;DP1;HR;Rf;'
modHalfInitNoCc = 'vv0;Xm247;Ca0.50;BZ;Ff0;DP1;HR;Rf;'
modEmptInitBB = 'vv0;Xm247;Ca0.090;BZ;Ff0;DP1;HR;Rf;'
modEmptInitCHG = 'vv0;Xm247;Ca-0.004;BZ;Ff0;DP1;HR;Rf;'
modEmptInitGen = 'vv0;Xm247;Ca0.17;BZ;Ff0;DP1;HR;Rf;'
noisePackage = 'DT.05;DV0.3;DM.75;DN6;'
silentPackage = 'DT;DV;DM;DN;'
synced_slow = 'Dr400;D>400;ED1;DP1;'
synced_slow_pulse = 'Dr800;D>800;ED1;DP1;'
slow = synced_slow_pulse
quiet = 'vv0;Dr;DP;D>;Dh;'
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
dmn50 = time_stamp + 'Dn-50;Dm-50;'
cmn50 = time_stamp + 'Dn50;Dm50;'
dmn100 = time_stamp + 'Dn-100;Dm-100;'
cmn100 = time_stamp + 'Dn100;Dm100;'
sc50 = time_stamp + 'DI50;'  # 50 amp discharge
sd50 = time_stamp + 'DI-50;'  # 50 amp discharge
c00 = 'Pf;W2;Dm;Dn;Rf;W50;'
dv0 = 'Pf;W2;Dv;Rf;W50;'
s00 = 'Pf;W2;DI;Rf;W100;'
twitch = time_stamp + 'XR;'
vm12 = 'Dv-12;'

# Note:  Photon 2 is throughput limited on the Serial buses.  The *tweak* transients are sensitive to differences
# caused by over-runs and slip and set Dr400 before Xp* then resets to Dr100 (nominal).
lookup = {
        'satInit': (22, 'Y;RS;RV;' + quiet + 'cc;Dh;Dr;*W;*vv0;*XS;*Ca1;BZ;Ff0;DP1;<HR;<Rf;<XK;', ('',)),
        'initMid': (22, 'Y;RS;RV;' + quiet + 'cc;Dh1800000;*W;*vv0;*XS;*Ca.5;BZ;Ff0;<HR;<Rf;<XK;', ('',)),
        'saveAdjusts': (60, 'vv4;Dh1000;PR;PV;Pr;Pr;BP2;Pr;BP;Pr;BS2;Pr;BS;Pr;Pr;Pr;DA5;Pr;DB-5;Pr;RS;Pr;Dc0.2;Pr;Dc;DI-10;Pr;DI;Pr;Dt5;Pr;Dt;Pr;SA2;Pr;SA;Pr;SB2;Pr;SB;Pr;si-1;Pr;RS;Pr;Sk2;Pr;Sk;Pr;SQ2;Pr;SQ;Pr;Sq3;Pr;Sq;Pr;SV1.1;Pr;SV;Pr;Xb10;Pr;Xb;Pr;Xa1000;Pr;Xa;Pr;Xf1;Pr;RS;Pr;Xm10;Pr;RS;Pr;W3;vv0;XQ3;PR;PV;XQ60000;Dh;<XD;', ("For testing out the adjustments and memory", "Read through output and witness set and reset of all", "The DS2482 moderate headroom should not exceed limit printed.  EG 11 of 12 is ok.")),
        'custom': (72, 'XQ60000;<XD;', ("For general purpose data collection", "'save data' will present a choice of file name", "")),
        'allInBB': (1200,
                    slow + 'Dh4000;' +
                    modEmptInitBB + slowTwitchDef + 'Xa-162;' + slowTranPrep + twitch + 'XQ568000;' + 'Xa;' + tempCleanup +  # offSitBmsBB
                    'Xm247;Ca0.9962;' + fastTwitchDef + 'Xa17;' + slowTranPrep + 'XR;XQ600000;' + 'Xa;' +  # satSitBB
                    quiet + cleanup + '<XD;',
                    ('All the best transients BB', "Must have same 'vv*' throughout", "")),
        'doNothing': (34, 'vv0;DP1;vv4;W50;vv0;W2;Hd;<XD;', ("Do nothing", "--", "--", "--")),
        'zero_with_pc': (120, hdwNoVbPcMidInit + zeroPrepHdweNoVb + 'vv4;W17;' + 'XQ25000;' + 'vv99;Xm2;XQ15000;' + quiet  + cleanup + '<XD;', ("Hardware zero_with_pc run", "", "", "")),
        # Begin actual regression cases here
        'ampHiEmptFail': (130, modLoInit + tranPrep + c50 + 'XQ25000;' + c00 + quiet + cleanup + '<XD;', ("Inject 50A into amp.  Should detect and switch amp current failure", "'diff' will be displayed. After a bit more, current display will change to 0.", "To evaluate plots, start looking at 'Ult 1'. Fault record (frozen). Will see 'diff' flashing on display soon after fault cleared automatically (lost redundancy).  Also will see verification imbedded model respond to the bad current signal by elevating vb, an effect that won't appear in data from app.", "Loss of ibm set 'accy' because loss of most accurate sensor.")),
        'ampHiFail': (130, modHalfInit + tranPrep + c50 + 'XQ25000;' + c00 + quiet + cleanup + '<XD;', ("Inject 50A into amp.  Should detect and switch amp current failure", "'diff' will be displayed. After a bit more, current display will change to 0.", "To evaluate plots, start looking at 'Ult 1'. Fault record (frozen). Will see 'diff' flashing on display soon after fault cleared automatically (lost redundancy).  Also will see verification imbedded model respond to the bad current signal by elevating vb, an effect that won't appear in data from app.", "Loss of ibm set 'accy' because loss of most accurate sensor.")),
        'noaHiFail': (130, modHalfInit + tranPrep + d50 + 'XQ25000;' + c00 + quiet + cleanup + '<XD;', ("Inject 50A into amp. With ib_diff only nothing changes then should isolate to the noa by wrap and choose amp.", "'diff' will be displayed then ib_fail due to wrap of noa", "To evaluate plots, start looking at 'Ult 1'. Fault record (frozen).", "Loss of ib set 'accy' because loss of current sensing at high currents.")),
        'rapidTweakRegression': (230, slow + 'Rs;W8;Xp10;' + quiet + cleanup + '<XD;', ('Should run three very large current discharge/recharge cycles without latched fail', 'Best test for seeing time skews and checking fault logic for false trips', 'Occasional jumps in ib_sel_stat are normal when pass through 0 A.  And Noa will fault and fail temprorarily')),
        'allProto': (512, modHalfInit + tranPrep + c50 + 'XQ25000;' + c00 + tempCleanup + '  Rs;W4;Xp10;  Rs;W4;Xp13;  ' + modHalfInitNoCc + tranPrep + cm50 + 'XQ50000;' + c00 + quiet + cleanup + '<XD;', ('Proto multi', "Must have same 'vv*' throughout", "No 'HR' either")),
        'pulseSoft': (85, synced_slow_pulse + 'XS;Dm0;Dn0;vv0;Xm255;Ca.5;Pm;W2;Rs;W20;vv4;W10;' + 'Xp7;W10;Pc;' + quiet + cleanup + '<XD;', ("Should generate a very short <10 sec data burst with a current sensor pulse.  Look at plots for good overlay. e_wrap should be nearly flat after a pulse response.", "This is the shortest of all tests.  Also useful for quick check tests.", "")),
        'pulseHard': (85, synced_slow_pulse + 'XS;Dm0;Dn0;vv0;Xm255;Ca.5;Pm;W2;Rs;W20;vv4;W10;' + 'Xp8;W10;Pc;' + quiet + cleanup + '<XD;', ("Should generate a very short <10 sec data burst with a hardware current pulse.  Look at plots for good overlay. e_wrap should be flat.", "This is the shortest of all tests.  Also useful for quick check tests.", "")),
        'rapidTweakRegressionH0': (230, 'Sh0;' + slow + 'Rs;W4;Xp10;Pf;W2;' + quiet + cleanup + '<XD;', ('Should run three very large current discharge/recharge cycles without fault', 'No hysteresis. Best test for seeing time skews and checking fault logic for false trips', 'Tease out cause of e_wrap faults.  e_wrap MUST be flat!', 'Occasional jumps in ib_sel_stat are normal when pass through 0 A')),
        'offLowSoc': (160, modEmptInitGen + tranPrep  + vm12 + 'XQ55000;' + dv0 + quiet + cleanup + '<XD;', ('Test for clean faults on shutoff.',)),
        'offSitBmsBB': (670, modEmptInitBB + slowTwitchDef + 'Xa-162;' + tranPrep + twitch + 'XQ568000;' + 'Pf;W2;Xa0;' + quiet + cleanup + '<XD;', ('for CompareRunRun.py Argon vs Photon builds. This is the only test for that.',)),
        'offSitBmsCHG': (670, modEmptInitCHG + slowTwitchDef + 'Xa-324;' + tranPrep + twitch + 'XQ568000;' + 'Pf;W2;Xa0;' + quiet + cleanup + '<XD;', ('for CompareRunRun.py Argon vs Photon builds. This is the only test for that.',)),
        'triTweakDisch': (230, slow + 'Rs;W4;Xp13;' + quiet + cleanup + '<XD;', ('Should run three very large current discharge/recharge cycles without fault', 'Best test for seeing time skews and checking fault logic for false trips', 'Occasional jumps in ib_sel_stat are normal when pass through 0 A.  Also hyst evident in one _s model')),
        'ampHiFailFf': (150, modHalfInit + tranPrep + 'Ff1;' + c50 + 'XQ40000;' + c00 + quiet + cleanup + '<XD;', ("Should detect but not switch amp current failure. (See 'diff' and current!=0 on display).", "Run about 60s. Start by looking at 'Ult 1'. No fault record (keeps recording).  Verify that on Fig 3 the e_wrap goes through a threshold ~0.4 without change of 'ib_sel_stat'", "This show when deploy with Fake Faults (Ff) don't throw false trips (it happened)", "ib_amp limited by max range e.g. 12.6.  ib_diff_fa will set red_loss but wait for wrap_fa to isolate and make selection change")),
        'ampLoFail': (155, modHalfInit + tranPrep + cm50 + 'XQ50000;' + c00 + quiet + cleanup + '<XD;', ("Should detect and switch amp current failure.", "Start looking at 'Ult 1'. Fault record (frozen). Will see 'diff' flashing on display even after fault cleared automatically (lost redundancy).", "ib_diff_fa will set red_loss but wait for wrap_fa to isolate and make selection change")),
        'ampLoFullFail': (155, modFullInit + tranPrep + cm50 + 'XQ50000;' + c00 + quiet + cleanup + '<XD;', ("Should detect and switch amp current failure before saturation tripped (would only be a problem for noa).", "Start looking at 'Ult 1'. Fault record (frozen). Will see 'diff' flashing on display even after fault cleared automatically (lost redundancy).", "ib_diff_fa will set red_loss but wait for wrap_fa to isolate and make selection change")),
        'noaLoFail': (155, modHalfInit + tranPrep + dm50 + 'XQ50000;' + c00 + quiet + cleanup + '<XD;', ("Should detect and switch amp current failure.", "Start looking at 'Ult 1'. Fault record (frozen). Will see 'diff' flashing on display even after fault cleared automatically (lost redundancy).", "ib_diff_fa will set red_loss but wait for wrap_fa to isolate and make selection change")),
        'noaLoFullFail': (155, modFullInit + 'DS-0.30' + tranPrep + dm50 + 'XQ50000;' + c00 + quiet + cleanup + '<XD;', ("Race with artificially low SAT logic to detect and switch amp current failure.", "Start looking at 'Ult 1'. Fault record (frozen). Will see 'diff' flashing on display.", "ib_diff_fa will set red_loss but wait for wrap_fa to isolate and make selection change")),
        'ampHiFailNoise': (140, modHalfInit + tranPrep + noisePackage + c50 + 'XQ25000;' + c00 + silentPackage + quiet + cleanup + '<XD;', ("Noisy ampHiFail.  Should detect and switch amp current failure.", "Start looking at 'Ult 1'. Fault record (frozen). Will see 'diff' flashing on display even after fault cleared automatically (lost redundancy).", "ib_diff_fa will set red_loss but wait for wrap_fa to isolate and make selection change")),
        'noaHiFailNoise': (140, modHalfInit + tranPrep + noisePackage + d50 + 'XQ25000;' + c00 + silentPackage + quiet + cleanup + '<XD;', ("Noisy ampHiFail.  Should detect and switch amp current failure.", "Start looking at 'Ult 1'. Fault record (frozen). Will see 'diff' flashing on display even after fault cleared automatically (lost redundancy).", "ib_diff_fa will set red_loss but wait for wrap_fa to isolate and make selection change")),
        'rapidTweakRegression40C': (220, 'D^15;' + slow + 'Rs;W4;Xp10;' + quiet + cleanup + '<XD;', ("Should run three very large current discharge/recharge cycles without fault", "Self-terminates", 'Occasional jumps in ib_sel_stat are normal when pass through 0 A')),
        'slowTweakRegression': (700, slow + 'Rs;W4;Xp11' + quiet + cleanup + '<XD;', ("Should run one very large slow (~15 min) current discharge/recharge cycle without fault.   It will take 60 seconds to start changing current.", 'Occasional jumps in ib_sel_stat are normal when pass through 0 A')),
        'satSitBB': (690, 'Xm247;Ca0.9962;' + fastTwitchDef + 'Xa17;' + tranPrep + 'XR;XQ600000;' + 'Xa0;' + quiet + cleanup + '<XD;', ("Should run one saturation and de-saturation event without fault.   Takes about 15 minutes.", "operate around saturation, starting below, go above, come back down. Tune Ca to start just below vsat",)),
        'satSitCHG': (690, 'Xm247;Ca0.986;' + fastTwitchDef + 'Xa17;' + tranPrep + 'XR;XQ600000;' + 'Xa0;' + quiet + cleanup + '<XD;', ("Should run one saturation and de-saturation event without fault.   Takes about 15 minutes.", "operate around saturation, starting below, go above, come back down. Tune Ca to start just below vsat",)),
        'flatSit': (690, 'Xm247;Ca0.9;Rb;Rf;Xts;Xa-81;Xf0.004;XW10000;XT10;XC2;W1;' + tranPrep + 'XR;XQ580000;Xa0;Xb0;' + quiet + cleanup + '<XD;', ("Operate around 0.9.  For CHINS, will check EKF with flat voc(soc).   Takes about 10 minutes.", "Make sure EKF soc (soc_ekf) tracks actual soc without wandering.")),
        'offSitBmsNoiseBB': (670, modEmptInitBB + slowTwitchDef + 'Xa-162;' + noisePackage + tranPrep + 'XR;XQ568000;' + 'Xa0;' + silentPackage + quiet + cleanup + '<XD;', ("Stress test with 2x normal Vb noise DV0.10.  Takes about 10 minutes.", "operate around saturation, starting above, go below, come back up. Tune Ca to start just above vsat. Go low enough to exercise hys reset ", "Make sure comes back on.", "It will show one shutoff only since becomes biased with pure sine input with half of down current ignored on first cycle during the shutoff.")),
        'offSitBmsNoiseCHG': (670, modEmptInitCHG + slowTwitchDef + 'Xa-324;' + noisePackage + tranPrep + 'XR;XQ568000;' + 'Xa0;' + silentPackage + quiet + cleanup + '<XD;', ("Stress test with 2x normal Vb noise DV0.10.  Takes about 10 minutes.", "operate around saturation, starting above, go below, come back up. Tune Ca to start just above vsat. Go low enough to exercise hys reset ", "Make sure comes back on.", "It will show one shutoff only since becomes biased with pure sine input with half of down current ignored on first cycle during the shutoff.")),
        # TODO for all volatile and saved parameters in Battery.csv:  'tranPrep' no 'vv' statment.  'stream' starts data incl 'vv'.  All adjusts before 'stream'
        'ampHiFailSlow': (535, modHalfInit + 'Fc0.0006;Fd0.5;' + tranPrep + c10 + 'XQ400000;' + c00 + quiet + cleanup + '<XD;', ("10A bias on amp, disable wrap, noa in range at 0A and reflects battery state. Artificially tight cc_diff threshold.  Will detect diff but no wrap. Will be slow (~6 min) cc_diff detection as it waits for the EKF to wind up to produce a cc_diff fault and complete isolation and switch to noa.", "EKF should tend to follow voltage while soc wanders away.", "Run for 6  minutes to see that cc_diff_fa does set")),
        'noaHiFailSlow': (525, modHalfInit+ 'Fc0.0006;' + tranPrep + d20 + 'XQ400000;' + c00 + quiet + cleanup + '<XD;', ("20A bias on noa, amp in range at 0A and reflects battery state. Artificially tight cc_diff threshold. Will detect and switch noa current failure due to wrap+diff. Once wrap trips diff won't be displayed. Cannnot ever produce a cc_diff fault because amp still used.", "Will display “diff” due to 20A difference.", "EKF won't move because fed by amp.", "Run for 6  minutes to verify not cc_diff_fa")),
        'noaHiFailSlower': (525, modHalfInit+ 'Fc0.0006;' + tranPrep + d08 + 'XQ400000;' + c00 + quiet + cleanup + '<XD;', ("8A bias on noa, amp in range at 0A and reflects battery state.  Artificially tight cc_diff threshold. Will detect and switch noa current failure due to wrap+diff. Once wrap trips diff won't be displayed. Cannnot ever produce a cc_diff fault.", "Will display “diff” due to 6 A difference..", "EKF won't move because fed by amp.", "Run for 6  minutes to see potential cc_diff_fa")),
        'noaHiFailSlowest': (525, modHalfInit+ 'Fc0.0006;' + tranPrep + d05 + 'XQ400000;' + c00 + quiet + cleanup + '<XD;', ("5A bias on noa, amp in range at 0A and reflects battery state. Artificially tight cc_diff threshold. Not enough current to trip the noa wrap.  Cannnot ever produce a cc_diff fault because amp still used.", "Will display “diff” due to 5 A difference..", "EKF won't move because fed by amp.", "Run for 6  minutes to see potential cc_diff_fa")),
        'vHiFail': (190, modHalfInit + tranPrep + 'XY;Dv0.82;XQ60000;' + dv0 + quiet + cleanup + '<XD;', ("Should detect voltage failure and display '*fail' and 'redl' within 60 seconds.", "To diagnose, begin with 'Ult 1'.   Look for e_wrap to go through ewlo_thr.", "You may have to increase magnitude of injection (Dv).  The threshold is 32 * r_ss.", "There MUST be no SATURATION")),
        'vHiFailNoise': (165, modHalfInit + noisePackage + tranPrep + 'XY;Dv0.82;XQ60000;' + dv0 + quiet + cleanup + '<XD;', ("Should detect voltage failure and display '*fail' and 'redl' within 60 seconds.", "To diagnose, begin with 'Ult 1'.   Look for e_wrap to go through ewlo_thr.", "You may have to increase magnitude of injection (Dv).  The threshold is 32 * r_ss.", "There MUST be no SATURATION")),
        'vHiFailFf': (180, modHalfInit + tranPrep + 'Ff1;XY;Dv0.8;XQ60000;' + dv0 + quiet + cleanup + '<XD;', ("Run for about 1 minute.", "Should detect voltage failure (see DOM1) but not display anything on display.", "Usually shows SAT.")),
        'pulseSSH': (-15, synced_slow + 'Xp8;' + quiet + cleanup + '<XD;', ("Should generate a very short <10 sec data burst with a hw pulse.  Look at plots for good overlay. e_wrap should be flat.", "This is the shortest of all tests.  Useful for quick checks.", "ib_diff_flt will take time beyond event to reset running Hi-Lo.")),
        'tLoFailModel': (275, modHalfInit + 'D^7;' + tranPrep + 'XY;W10;D^-113;XQ120000;' + 'D^;Rf;W50;' + cleanup + '<W50;' + quietwait + '<Pf;<XD;', ("Simulates open thermistor.", "To diagnose, begin with 'Ult 1'.   Look for e_wrap to go through ewlo_thr.", "You may have to increase magnitude of injection (Dv).  The threshold is 32 * r_ss.", "There MUST be no SATURATION")),
        'tHiFailModel': (275, modHalfInit + 'D^7;' + tranPrep + 'XY;W10;D^+50;XQ120000;' + 'D^;Rf;W50;' + cleanup + '<W50;' + quietwait + '<Pf;<XD;', ("Simulates open thermistor.", "To diagnose, begin with 'Ult 1'.   Look for e_wrap to go through ewlo_thr.", "You may have to increase magnitude of injection (Dv).  The threshold is 32 * r_ss.", "There MUST be no SATURATION")),
        'tLoFailHdwe': (275, modHalfInit230 + tranPrep + 'XY;W10;Dt-113;XQ120000;' + 'Dt;Rf;W50;' + cleanup + '<W50;' + quietwait + '<Pf;<XD;', ("Simulates open thermistor.", "To diagnose, begin with 'Ult 1'.   Look for e_wrap to go through ewlo_thr.", "You may have to increase magnitude of injection (Dv).  The threshold is 32 * r_ss.", "There MUST be no SATURATION")),
        'tHiFailHdwe': (275, modHalfInit230 + tranPrep + 'XY;W10;Dt+50;XQ120000;' + 'Dt;Rf;W50;' + cleanup + '<W50;' + quietwait + '<Pf;<XD;', ("Simulates open thermistor.", "To diagnose, begin with 'Ult 1'.   Look for e_wrap to go through ewlo_thr.", "You may have to increase magnitude of injection (Dv).  The threshold is 32 * r_ss.", "There MUST be no SATURATION")),
        'stepDown': (165, modHalfInit + tranPrep + sd50 + 'XQ25000;' + s00 + quiet + cleanup + '<XD;', ("Should be normal hard discharge step", "", "", "")),
        'stepUp': (165, modHalfInit + tranPrep + sc50 + 'XQ25000;' + s00 + quiet + cleanup + '<XD;', ("Should be normal hard charge step", "", "", "")),
        'ibDualMid': (130, modHalfInit + tranPrep + cmn100 + 'XQ25000;' + c00 + quiet + cleanup + '<XD;', ("Inject 100A into amp and noa simultaneously.  Should detect and switch dual amp current failure set Ib=0", "'*fail' will be displayed. After a bit more, current display will change to 0.", "To evaluate plots, start looking at 'Ult 1'. Fault record (frozen).")),
        'ibDualFlat': (130, modFlatInit + tranPrep + cmn100 + 'XQ25000;' + c00 + quiet + cleanup + '<XD;', ("Inject 100A into amp and noa simultaneously.  Should detect and switch dual amp current failure set Ib=0", "'*fail' will be displayed. After a bit more, current display will change to 0.", "To evaluate plots, start looking at 'Ult 1'. Fault record (frozen).")),
        'vcFlat': (130, modFlatInitHi + tranPrep + 'D30.6;' + 'XQ25000;' + 'Pf;W2;D3;Rf;W50;' + quiet + cleanup + '<XD;', ("Inject 0.6V into sensed Vc (normally 1.65).  Should fail both currents.", "To evaluate plots, start looking at 'Ult 1'. Fault record (frozen).")),
        }

# Lookup keys for which compare_run_sim should be called with shift_soc_s=False
no_shift_soc_s = frozenset({
    'rapidTweakRegression',
    'rapidTweakRegressionH0',
    'rapidTweakRegression40C',
    'triTweakDisch',
})

macro_lookup = {
        'end_early': (22, 'Y;cc;Dh1800000;*W;*vv0;*XS;*Ca1;<Hd;<Pf;', ('', '', '', '')),
        'hdwNoVbPcMidInit': (5, hdwNoVbPcMidInit, ('', '', '', '')),
        'modFullInit': (5, modFullInit, ('', '', '', '')),
        'modLoInit': (5, modLoInit, ('', '', '', '')),
        'modHalfInit': (5, modHalfInit, ('', '', '', '')),
        'modHalfInit230': (5, modHalfInit230, ('', '', '', '')),
        'modHalfInit239': (5, modHalfInit239, ('', '', '', '')),
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
        'cmn50': (5, cmn50, ('', '', '', '')),
        'dmn50': (5, dmn50, ('', '', '', '')),
        'cmn100': (5, cmn100, ('', '', '', '')),
        'dmn100': (5, dmn100, ('', '', '', '')),
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


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)

    def flush(self):
        for s in self._streams:
            s.flush()


def add_to_clip_board(text):
    pyperclip.copy(text)


# Begini - configuration class using .ini files
class Begini(ConfigParser):

    def __init__(self, name, default_dict_):
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
                self.read_dict(default_dict_)
                self.write(cfg_file)
            print('wrote', self.config_file_path)

    def get_item(self, ind, item):
        return self[ind][item]

    def put_item(self, ind, item, value):
        self[ind][item] = value
        self.save_to_file()

    def save_to_file(self):
        with open(self.config_file_path, 'w') as cfg_file:
            self.write(cfg_file)
        print('wrote', self.config_file_path)


def contain_all(testpath):
    folder_path, basename = str(PurePosixPath(testpath).parent), PurePosixPath(testpath).name
    parent, txt = str(PurePosixPath(folder_path).parent), PurePosixPath(folder_path).name
    key = ''
    with open(testpath, 'r') as file:
        for line in file:
            if line.__contains__(txt):
                shorter = line[line.find(txt):]
                end_key = shorter.find(',')
                key = shorter[:end_key].strip()
                break
    return folder_path, parent, basename, txt, key


def copy_clean(src, dst):
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    with open(src, 'r') as file_in:
        data = file_in.read()
    with open(dst, 'w') as file_out:
        file_out.write(data.replace('\0', ''))


def create_file_key(version_, unit_, battery_):
    return version_ + '_' + unit_ + '_' + battery_


def create_file_txt(option_, unit_, battery_):
    return option_ + '_' + unit_ + '_' + battery_ + '.csv'


def empty_file(target):
    try:
        with open(target, 'w') as _:
            pass
    except Exception as e:
        print(f"empty_file: failed to empty {target} with {e}")
    print('emptied', target)


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


last_task = None
last_task_args = ()
last_task_kwargs = {}


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


def size_of(path):
    if Path(path).is_file() and (size := Path(path).stat().st_size) > 0:  # bytes
        return size
    else:
        return 0
