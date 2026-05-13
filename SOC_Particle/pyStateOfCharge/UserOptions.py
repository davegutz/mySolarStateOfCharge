# UserOptions - argument list consolidation
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

""" General data-over-model plotting options
Dependencies:
    - SavedData  (structures)
"""

from SavedData import SavedData, SavedDataSim
from Battery import Battery as Battery
from dataclasses import dataclass
from typing import Optional


@dataclass
class UserOptions:
    mon_run: SavedData  # Mandatory reference data to be replicated
    run_type: Optional[str] = None  # Either "RunSim" or "HistSim" depending on caller
    sim_run: Optional[SavedDataSim] = None  # Embedded model data
    unit: Optional[str] = None  # Name of the battery instance derived from 'HDWE_UNIT' of configuration include .h file
    Bsim: Optional[int] = None  # sim model code BB=0 (Battleborn), CH=1 (Chins), CHG=2 (Chins in Garage)
    Bmon: Optional[int] = None  # mon model code BB=0 (Battleborn), CH=1 (Chins), CHG=2 (Chins in Garage)
    init_time: Optional[float] = -4.  # The process tries to determine mon_run.init_time when data is loaded by finding
    # when Ib changes. This input helps out to over-ride those results when they don't work as desired. It shouldn't
    # be needed often.
    max_time: Optional[float] = None  # Limit the simultation run, s

    # Model scalar / adders
    scale_batt: Optional[float] = None  # Battery size scalar applied to the nominal battery unit of 100 A-h
    slr_cap_chg: Optional[float] = 1.  # Scalar on ideal capacitor model for hysteresis charging model only
    slr_cap_dis: Optional[float] = 1.  # Scalar on ideal capacitor model for hysteresis discharging model only
    slr_coul_eff: Optional[float] = 1.  # Scalar on Coulombic Efficiency of battery model, both for the BatterySim model
    # and the BatteryMonitor Coulomb counter
    slr_cutback_gain: Optional[float] = 1.  # Scalar on the automatic BatterySim model of saturation effects
    slr_hys_cap_sim: Optional[float] = 1.  # Scalar on the battery size effect on hysteresis
    slr_hys_chg: Optional[float] = 1.  # Direct scalar on the magnitude of hysteresis during charging
    slr_hys_dis: Optional[float] = 1.  # Direct scalar on the magnitude of hysteresis during charging
    slr_hys_mon: Optional[float] = 1.  # Overall scalar on the magnitude of hysteresis in BatteryMonitor
    slr_hys_sim: Optional[float] = 1.  # Overall scalar on the magnitude of hysteresis in BatterySim
    slr_res_0: Optional[float] = 1.  # Scalar on Randles static resistance model
    slr_res_ct: Optional[float] = 1.  # Scalar on Randles charge transfer function resistance
    slr_r_ss: Optional[float] = 1.  # Scalar on equivalent battery resistance state-space charge transfer
    # TODO: when is ss used versus ct
    slr_tauct_sim: Optional[float] = 1.  # Scalar on Randles charge transfer function time constant in ModelSim
    add_voc_sim: Optional[float] = 0.  # Adder to BatterySim voc table outputs (should match dvoc of Chemistry_BMS.cpp)
    add_voc_mon: Optional[float] = 0.  # Adder to BatteryMonitor voc table outputs (should match dvoc of

    # Failure injection
    ib_fail_t: Optional[float] = None  # Time to inject a failure into the Ib input signal
    ib_fail: Optional[float] = 0.  # The fixed Ib value to fail to, A
    vb_fail_t: Optional[float] = None  # Time to inject a failure into the Vb input signal
    vb_fail: Optional[float] = 13.2  # The fixed Vb value to fail to, V

    # Configuration changes
    eframe_mult: Optional[int] = Battery.ap_eframe_mult

    stauct_mon: Optional[float] = 1.
    use_vb_sim: Optional[bool] = False
    request_history: Optional[int] = 5  # Print simulation history (0 - 5) to check overplot using data in addition
    use_ib_mon: Optional[bool] = False  # Drive BatterySim directly with the BatteryMonitor input, useful when raw sim data not available
    use_sat_mon: Optional[bool] = False  # Drive entire model directly with the run input, useful for HistSim unable to accurately run sliding deadbanc
    use_mon_soc: Optional[bool] = False  # Drive SOC of the model directly with data to focus on modeling that is downstream of SOC
    use_vb_raw: Optional[bool] = False  # Force usage of raw Vb bypassing the signal selection logic
    verbose: Optional[bool] = True  # Lots of 'helpful' information used to provide some quick clues about whatever
    # to or instead of plots
    mod_force: Optional[int] = None  # Force modeling config that cannot be gleaned from input data or other reason
