# CompareRunHist.py:  combine a CompareRunSim with CompareHistSim
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

from CompareRunSim import compare_run_sim
from CompareHistSim import compare_hist_sim

# Suppress all UserWarning messages
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# noinspection PyUnusedLocal,PyPep8Naming
def compare_run_hist(data_file=None, unit_key=None, time_end=None, plots=True,
                     strict_overplot=False, terse=False, use_mon_csv=False, dt_resample=10, Tb_force=None,
                     use_mon_soc=False, verbose=False, scale=1., slr_hys_sim=1., Battery=None,
                     init_time=None, time_shift=None, mon_str='', sync_time=None,
                     request_history_run_sim=None, request_history_hist_sim=None):
    print(f"\n compare_run_hist: \
    \n{data_file=} \
    \n{unit_key=} \
    \n{time_end=} \
    \n{plots=} \
    \n{strict_overplot=} \
    \n{terse=} \
    \n{use_mon_csv=} \
    \n{dt_resample=} \
    \n{Tb_force=} \
    \n{use_mon_soc=} \
    \n{verbose=} \
    \n{scale=} \
    \n{slr_hys_sim=} \
    \n{init_time=} \
    \n{time_shift=} \
    \n{mon_str=} \
    \n{sync_time=} \
    \n{request_history_run_sim=} \
    \n{request_history_hist_sim=} \
    \n ")

    fig_list, fig_files = \
        compare_run_sim(data_file=data_file, unit_key=unit_key, plots=plots, time_shift=time_shift,
                        use_mon_soc_=use_mon_soc, verbose=verbose, scale_batt=scale, slr_hys_sim=slr_hys_sim,
                        request_history=request_history_run_sim, init_time=init_time,
                        strict_overplot=strict_overplot, terse=terse, show_killer_=False)

    _, _ = \
        compare_hist_sim(data_file=data_file, use_mon_csv=use_mon_csv, unit_key=unit_key, dt_resample=dt_resample,
                         plots=plots, Tb_force=Tb_force, request_history=request_history_hist_sim, terse=terse,
                         strict_overplot=strict_overplot, fig_list=fig_list, fig_files=fig_files, show_killer_=True)

    pass


# noinspection PyPep8Naming
def main():  # Example usage:  ok 20260217

    # Cut-pasted from GUI_TestSOC Run window
    data_file = 'G:/My Drive/GitHubArchive/SOC_Particle/dataReduction\\g20250612a\\ampHiEmptFail_soc3p2_hi_lo_bb.csv'
    unit_key = 'g20250612a_soc3p2_hi_lo_bb'
    time_end = None
    plots = True
    strict_overplot = True
    terse = True
    dt_resample = 10
    Tb_force = None
    use_mon_soc = False
    verbose = True
    request_history_run_sim = None
    request_history_hist_sim = None

    compare_run_hist(data_file=data_file, unit_key=unit_key, plots=plots, time_end=time_end,
                     use_mon_soc=use_mon_soc, verbose=verbose, strict_overplot=strict_overplot, terse=terse,
                     dt_resample=dt_resample, Tb_force=Tb_force,
                     request_history_run_sim=request_history_run_sim, request_history_hist_sim=request_history_hist_sim)

if __name__ == '__main__':  # Example usage.  Ran ok 202602xx
    main()
