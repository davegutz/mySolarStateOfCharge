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
"""Utility to load data from csv files"""
from CompareFault import add_stuff_f, filter_Tb, IB_BAND
from SavedData import SavedData, SavedDataSim
from Battery import Battery, BatteryMonitor
from battery_constants import load_off_nominal_battery, apply_off_nominal_battery
from DataOverModel import write_clean_file
from Util import rename_all
from resample import remove_nan
import numpy as np


def find_sync(path_to_data):
    sync = []
    with open(path_to_data) as in_file:
        for line in in_file:
            if line.__contains__('SYNC'):
                sync.append(float(line.strip().split(',')[-1]))

    if not sync:
        sync = None
    else:
        sync = np.array(sync)
    return sync

def calculate_master_sync(ref, test):
    delta = np.maximum(ref, test)
    return delta

def remove_0T(d_ra, info):
    """Remove useless 0 Time elements"""
    condition = d_ra['time_ux'] >= 1746684850783./1000.
    filtered_data = d_ra[condition]
    num_removed = len(d_ra) - len(filtered_data)
    if num_removed > 0:
        print(f"\nremove_0T:  screened out {num_removed} rows from {info} with bad time_ux element\n", flush=True)
    return filtered_data


class SyncInfo:
    """Shift time arrays to synchronize two different data sets for CompareRunRun usage"""
    def __init__(self, sav_mon, sync=None):
        self.is_empty = False
        if sync is None or sav_mon is None:
            self.is_empty = True
            return
        self.time_mon = sav_mon.time
        self.sync_cTime = sync
        self.cTime = sav_mon.cTime
        self.time = sav_mon.time
        self.int_mon = []
        self.length = len(sync)
        rel = []
        delta = []
        for i in np.arange(self.length):
            rel.append(self.sync_cTime[i] - sav_mon.cTime[0])
            if i == 0:
                delta.append(rel[0])
                self.int_mon.append([np.where(sav_mon.cTime <= sync[i])])
            else:
                delta.append(rel[i] - rel[i-1])
                self.int_mon.append([np.where((sav_mon.cTime <= sync[i]) & (sav_mon.cTime > sync[i-1]))])
        self.int_mon.append([np.where(sav_mon.cTime > sync[self.length-1])])
        self.rel_mon = np.array(rel)
        self.del_mon = np.array(delta)
        return

    def synchronize(self, sync_del):
        """Call this after building two class instances and calling calculate_master_sync to make sync_del"""
        # Init entire time array again.  First sync is always 0
        acc_shift = self.sync_cTime[0]
        self.time_mon = self.cTime.copy()

        # Subsequent sets based on difference to master del
        for i in np.arange(self.length+1):
            if 1 < i:
                acc_shift -= sync_del[i-1] - self.del_mon[i-1]
            self.time_mon[self.int_mon[i]] = (self.time_mon[self.int_mon[i]] - acc_shift).copy()

        return


# Load from files
def load_data(path_to_data, skip, unit_key, zero_zero, time_end, rated_batt_cap=Battery.NOM_UNIT_CAP,
              legacy=False, zero_thr_in=0.02, init_time=None, time_shift=None, mon_str=''):

    print(f"load_data: \n{path_to_data=}\n{skip=}\n{unit_key=}\n{zero_zero=}\n{time_end=}\n{rated_batt_cap=}\n"
          f"{legacy=}\n{init_time=}\n{time_shift=}\n")

    battery_hdr = "Battery_hdr"
    battery_val = "Battery_val"
    hdr_key_rap = "unit_rap,"  # Find one instance of title
    hdr_key_sel = "unit_s,"  # Find one instance of title
    unit_key_sel = "unit_sel"
    hdr_key = "unit_e,"  # Find one instance of title
    unit_key_ekf = "unit_ekf"
    hdr_key_sim = "unit_m,"  # Find one instance of title
    unit_key_sim = "unit_sim"
    temp_flt_file = 'flt_compareRunSim.txt'
    hdr_key_shunt = "unit_shunt"
    unit_key_shunt = "shunt_unit"

    sync = find_sync(path_to_data)
    # Load battery (ref)
    battery_file_clean = write_clean_file(path_to_data, type_='_battery', hdr_key=battery_hdr,
                                          unit_key=battery_val, skip=skip)
    if battery_file_clean:
        battery_raw = np.genfromtxt(battery_file_clean, delimiter=',', names=True, dtype=float).view(np.recarray)
    else:
        battery_raw = None
        print(f"load_data: returning battery_raw=None")
    # Load off-nominal Battery values
    if battery_raw is not None:
        # Scroll through all off-nominals make dictionary
        Battery_off_dict = load_off_nominal_battery(Battery_to_add=battery_raw)
        apply_off_nominal_battery(Battery, Battery_off_dict)
    if Battery_off_dict is None:
        return None, None, None, None, None, None

    # Load fault
    batt = BatteryMonitor()
    temp_flt_file_clean = write_clean_file(path_to_data, type_='_flt', hdr_key='fltb',
                                           unit_key='unit_f', skip=skip, comment_str='---')
    if temp_flt_file_clean:
        f_raw = np.genfromtxt(temp_flt_file_clean, delimiter=',', names=True, dtype=float).view(np.recarray)
    else:
        print("data from", temp_flt_file, "empty after loading")
        f_raw = None
    if f_raw is not None:
        f_raw = np.unique(f_raw)
        f_raw = remove_nan(f_raw)
        f_raw = rename_all(f_raw)
        batt.sp_vsat_add = Battery_off_dict['sp_vsat_add']
        f = add_stuff_f(f_raw, batt, ib_band=IB_BAND, ap_ib_diff_slr=Battery_off_dict['ap_ib_diff_slr'],
                        ap_ib_quiet_slr=Battery_off_dict['ap_ib_quiet_slr'])
        # print("\nload_data:  f:\n", f, "\n")
        f = filter_Tb(f, 20., batt, tb_band=100., rated_batt_cap=rated_batt_cap)
        f.str = ''
    else:
        f = None
        print(f"load_data: returning f=None")

    data_file_clean = write_clean_file(path_to_data, type_='_mon', hdr_key=hdr_key_rap, unit_key=unit_key, skip=skip)
    if data_file_clean is None:
        print(f"load_data: returning mon=None")
        return None, None, f, None, temp_flt_file_clean, None
    mon_raw = np.genfromtxt(data_file_clean, delimiter=',', names=True, dtype=float).view(np.recarray)

    # Load sel (ref)
    sel_file_clean = write_clean_file(path_to_data, type_='_sel', hdr_key=hdr_key_sel,
                                      unit_key=unit_key_sel, skip=skip)
    if sel_file_clean:
        sel_raw = np.genfromtxt(sel_file_clean, delimiter=',', names=True, dtype=float).view(np.recarray)
        sel_raw = rename_all(sel_raw)
    else:
        sel_raw = None
        print(f"load_data: returning sel_raw=None")

    # Load ekf (ref)
    ekf_file_clean = write_clean_file(path_to_data, type_='_ekf', hdr_key=hdr_key,
                                      unit_key=unit_key_ekf, skip=skip)
    if ekf_file_clean:
        ekf_raw = np.genfromtxt(ekf_file_clean, delimiter=',', names=True, dtype=float).view(np.recarray)
        ekf_raw = rename_all(ekf_raw)
    else:
        ekf_raw = None
        print(f"load_data: returning ekf_raw=None")

    # Load shunt (ref)
    shunt_file_clean = write_clean_file(path_to_data, type_='_shunt', hdr_key=hdr_key_shunt,
                                      unit_key=unit_key_shunt, skip=skip)
    if shunt_file_clean:
        shunt_raw = np.genfromtxt(shunt_file_clean, delimiter=',', names=True, dtype=float).view(np.recarray)
    else:
        shunt_raw = None
        print(f"load_data: returning shunt_raw=None")

    mon = SavedData(battery=battery_raw, rap=mon_raw, sel=sel_raw, ekf=ekf_raw, shunt=shunt_raw,
                    time_end=time_end, zero_zero=zero_zero, zero_thr=zero_thr_in, sync_cTime=sync,
                    init_time=init_time, time_shift=time_shift, str_=mon_str)

    # Load sim _s v24 portion of real-time run (ref)
    data_file_sim_clean = write_clean_file(path_to_data, type_='_sim', hdr_key=hdr_key_sim,
                                           unit_key=unit_key_sim, skip=skip)
    if data_file_sim_clean:
        sim_raw = np.genfromtxt(data_file_sim_clean, delimiter=',', names=True, dtype=float).view(np.recarray)
        sim_raw = rename_all(sim_raw)
        sim = SavedDataSim(time_run_start=mon.time_run_start, data=sim_raw, time_end=time_end)
    else:
        sim_raw = None
        sim = SavedDataSim(time_run_start=mon.time_run_start, data=sim_raw, time_end=time_end, fake=True,
                           mon_for_fake=mon, str_='run_s')
        sim = rename_all(sim)
        print(f"load_data: returning sim=None")

    # Calculate sync information
    sync_info = SyncInfo(sav_mon=mon, sync=sync)

    return mon, sim, f, data_file_clean, temp_flt_file_clean, sync_info
