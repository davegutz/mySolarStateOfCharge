# CompareRunHist.py:  combine a CompareRunSim with CompareHistSim
# Copyright (C) 2024 Dave Gutz
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

import matplotlib.pyplot as plt
from CompareRunSim import compare_run_sim
from CompareHistSim import compare_hist_sim

import sys
if sys.platform == 'darwin':
    import matplotlib
    matplotlib.use('tkagg')
plt.rcParams['axes.grid'] = True
plt.rcParams['legend.fontsize'] = 'small'

# Suppress all UserWarning messages
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# noinspection PyUnusedLocal
def compare_run_hist(data_file=None, unit_key=None, time_end=None, plots=True,
                     strict_overplot=False, terse=False, use_mon_csv=False, dt_resample=10, Tb_force=None,
                     use_mon_soc=False, verbose=True, scale=1., slr_hys_sim=1., Battery=None,
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

    fig_list, fig_files =\
        compare_run_sim(data_file=data_file, unit_key=unit_key, plots=plots, time_end_in=time_end,
                        use_mon_soc_=use_mon_soc, verbose=verbose, scale_in=scale, slr_hys_sim=slr_hys_sim,
                        request_history=request_history_run_sim, init_time_in=init_time, time_shift_in=time_shift,
                        strict_overplot=strict_overplot, terse=terse, show_killer_=False)

    _, _ = \
        compare_hist_sim(data_file=data_file, use_mon_csv=use_mon_csv, unit_key=unit_key, dt_resample=dt_resample,
                         plots=plots, Tb_force=Tb_force, request_history=request_history_hist_sim, terse=terse,
                         strict_overplot=strict_overplot, fig_list=fig_list, fig_files=fig_files, show_killer_=True)

    pass


def main():
    data_file = 'G:/My Drive/GitHubArchive/SOC_Particle/dataReduction\\g20250612a\\ampHiFail_soc3p2_hi_lo_bb.csv'
    unit_key = 'g20250612a_soc3p2_hi_lo_bb'
    time_end = None
    # plots = False
    plots = True
    strict_overplot = False
    # terse = True
    terse = False
    use_mon_csv = False
    dt_resample = 1
    Tb_force = None
    use_mon_soc_ = False
    verbose = True
    use_mon_soc = False
    scale_in = 1.0
    slr_hys_sim = 1.0
    init_time_in = None
    time_shift_in = None
    mon_str = ''
    sync_time = None

    # RunSim plot selection
    # 1=ekf   2=soc  3=soc_s  4=temp   5=volt  6=ekf   7=dyn_m  8=vb_wrap
    request_hist_run_sim = 3
    # request_hist_run_sim = None

    # HistSim plot selection
    # 3=soc_s   5=volt
    request_hist_hist_sim = 5
    # request_hist_hist_sim = None

    compare_run_hist(data_file=data_file, unit_key=unit_key, plots=plots, time_end=time_end,
                     use_mon_soc=use_mon_soc, verbose=verbose, strict_overplot=strict_overplot, terse=terse,
                     dt_resample=dt_resample, Tb_force=Tb_force,
                     request_history_run_sim=request_hist_run_sim, request_history_hist_sim=request_hist_hist_sim)

if __name__ == '__main__':
    main()
