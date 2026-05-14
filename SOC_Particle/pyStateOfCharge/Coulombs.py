# BatteryEKF - general purpose battery class for embedded EKF
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

"""Coulomb Counting.  1 Ampere is 1 Coulombs/second charge rate of change.   Herein are various methods to keep track
of the totals and standardize the calculations."""

# Constants
from Chemistry_BMS import Chemistry
import Globals as G


class Coulombs:
    """Coulomb Counting"""

    def __init__(self, OPT=None, q_cap_rated=None, q_cap_rated_scaled=None, t_rated=None, tweak_test=False, mod_code=0,
                 dvoc=0., Dw=0.):
        if OPT is not None and hasattr(OPT, 'unit'):
            unit = OPT.unit
        else:
            unit = ''
        self.q_cap_rated = q_cap_rated
        self.q_cap_rated_scaled = q_cap_rated_scaled
        self.t_rated = t_rated
        self.delta_q = 0.
        self.d_delta_q = 0.
        self.q = 0.
        self.q_capacity = 0.
        self.soc_min = 0.
        self.soc = 1.
        self.resetting = True
        self.q_min = 0.
        self.sat = True
        self.saturated = True
        self.chm = mod_code
        self.tweak_test = tweak_test
        self.reset = False
        self.Tb_f = 0.
        self.chemistry = Chemistry(mod_code=mod_code, dvoc=dvoc, unit=unit, Dw=Dw)
        self.chemistry.assign_all_mod(mod_code, unit=unit)

    def __str__(self, prefix=''):
        """Returns representation of the object"""
        s = prefix + "Coulombs:\n"
        s += "  q_cap_rated = {:9.1f}    // Rated capacity at t_rated_, saved for future scaling, C\n"\
            .format(self.q_cap_rated)
        s += "  q_cap_rated_scaled = {:9.1f} // Applied rated capacity at t_rated_, after scaling, C\n"\
            .format(self.q_cap_rated_scaled)
        s += "  q_capacity = {:9.1f}     // Saturation charge at temperature, C\n".format(self. q_capacity)
        s += "  q =          {:9.1f}     // Present charge available to use, C\n".format(self. q)
        s += "  delta_q      {:9.1f}     // Charge since saturated, C\n".format(self. delta_q)
        s += "  soc =        {:7.3f}       // Fraction of saturation charge (q_capacity_) available (0-1)  soc)\n"\
            .format(self.soc)
        s += "  sat =          {:1.0f}          // Indication from caller that battery is saturated, T=saturated\n"\
            .format(self.sat)
        s += "  t_rated =    {:5.1f}         // Rated temperature, deg C\n".format(self.t_rated)
        s += "  Tb_f    =    {:5.1f}         // Rated temperature, deg C\n".format(self.tb_f)
        # s += "  temp_rlim =     {:7.3f}       // Tbatt rate limit, deg C / s\n".format(self.temp_rlim)
        s += "  resetting =     {:d}          // Flag to test coulomb counters, T = external reset of counter\n"\
            .format(self.resetting)
        s += "  soc_min =    {:7.3f}       // Lowest soc for power delivery.   Arises with temp < 20 C\n"\
            .format(self.soc_min)
        s += "  tweak_test =    {:d}          // Driving signal injection completely using software inj_soft_bias\n"\
            .format(self.tweak_test)
        s += "  coul_eff =   {:8.4f}      // Coulombic efficiency- fraction of charging turned into usable Coulombs\n".\
            format(self.chemistry.coul_eff)
        return s

    def apply_cap_scale(self, scale):
        """ Scale size of battery and adjust as needed to preserve delta_q.  Tb unchanged.
        Goal is to scale battery and see no change in delta_q on screen of
        test comparisons.   The rationale for this is that the battery is frequently saturated which
        resets all the model parameters.   This happens daily.   Then both the model and the battery
        are discharged by the same current so the delta_q will be the same."""
        self.q_cap_rated_scaled = scale * self.q_cap_rated
        self.q_capacity = self.calculate_capacity(tb_f=self.Tb_f)
        self.q = self.delta_q + self.q_capacity  # preserve self.delta_q, deficit since last saturation(like real life)
        if self.q_capacity != 0. and self.q_capacity is not None:
            self.soc = self.q / self.q_capacity
        self.resetting = True  # momentarily turn off saturation check

    # Memory set, adjust bookkeeping as needed.delta_q, capacity, temp preserved void
    def apply_delta_q_brief(self, delta_q):
        self.delta_q = delta_q
        self.q = self.delta_q + self.q_capacity
        if self.q_capacity != 0. and self.q_capacity is not None:
            self.soc = self.q / self.q_capacity
        self.resetting = True  # momentarily turn off saturation check

    def apply_delta_q_t(self, delta_q, tb_f):
        self.delta_q = delta_q
        self.Tb_f = tb_f
        self.q_capacity = self.calculate_capacity(tb_f=self.Tb_f)
        self.q = self.q_capacity + self.delta_q
        if self.q_capacity != 0. and self.q_capacity is not None:
            self.soc = self.q / self.q_capacity
        self.resetting = True

    def apply_soc(self, soc, tb_f, delta_q):
        """Memory set, adjust bookkeeping as needed.  delta_q preserved"""
        self.soc = soc
        self.Tb_f = tb_f
        self.q_capacity = self.calculate_capacity(tb_f=tb_f)
        self.q = self.soc * self.q_capacity
        # self.q_eps = delta_q + self.q_capacity * (1. - self.soc)
        # self.delta_q = -self.q_capacity * (1. - self.soc) + self.q_eps
        self.delta_q = delta_q
        self.resetting = True  # momentarily turn off saturation check

    def calculate_capacity(self, tb_f):
        """Capacity"""
        try:
            res = self.q_cap_rated_scaled * (1. + self.chemistry.dqdt * (tb_f - self.chemistry.rated_temp))
            if self.reset:
                pass
            else:
                pass
        except IOError:
            res = 1
        return res

    def count_coulombs(self, OPT, chem, dt, reset, tb_f, charge_curr, sat, saturated):
        """Count coulombs based on true=actual capacity
        Inputs:
            dt              Integration step, s
            tb_f            Battery temperature, deg C  (filtered usually to reduce electrical noise artifacts)
            charge_curr     Charge, A
            sat             Indicator that battery is saturated (VOC>threshold(temp)), T/F
            coul_eff        Coulombic efficiency - the fraction of charging input that gets turned into usable Coulombs
            use_mon_soc     Command to drive integrator with input mon_soc
            soc             Auxiliary integrator setting, fraction soc
        """
        if self.chm != chem:
            self.chemistry.assign_all_mod(chem)
            self.chm = chem
        self.reset = reset
        self.d_delta_q = charge_curr * dt
        if charge_curr > 0. and not self.tweak_test:
            self.d_delta_q *= self.chemistry.coul_eff
        self.sat = sat
        self.saturated = saturated
        self.Tb_f = tb_f

        # if charge_curr < 0.:
        #     print(f"{OPT.mon_run.time[G.i]=} {OPT.mon_run.time[G.i]-OPT.mon_run.time[G.i-1]=} {OPT.mon_run.dt[G.i]=} {dt} {OPT.mon_run.ib_charge[G.i]=} {charge_curr} {OPT.mon_run.ib_charge[G.i] * OPT.mon_run.dt[G.i]=} {dt * charge_curr} {OPT.mon_run.d_delta_q[G.i]=} {self.d_delta_q}")

        # Saturation.   Goal is to set q_capacity and hold it so remembers last saturation status.
        if self.saturated:
            if self.d_delta_q > 0:
                self.d_delta_q = 0.
                if ~self.resetting:
                    self.delta_q = 0.
        self.resetting = False  # one pass flag.  Saturation debounce should reset next pass

        # Integration
        self.q_capacity = self.calculate_capacity(tb_f=self.Tb_f)
        if OPT.use_mon_soc:
            self.soc = OPT.mon_run.soc[G.i]
            self.q = self.q_capacity * self.soc
            self.delta_q = self.q - self.q_capacity
        else:
            if not self.resetting and not self.reset:
                # self.delta_q = max(min(self.delta_q + d_delta_q - self.chemistry.dqdt*self.q_capacity*tb_f_rate*dt,
                #                        0.0), -self.q_capacity*1.5)
                #  Because delta_q is off of saturation and capacity book-kept elsewhere for soc, don't need to book
                #  the temperature effect here
                self.delta_q = max(min(self.delta_q + self.d_delta_q, 0.0), -self.q_capacity*1.5)

                self.q = self.q_capacity + self.delta_q
                if self.delta_q < -100.:
                    pass

        # Normalize
        if self.q_capacity != 0. and self.q_capacity is not None:
            self.soc = self.q / self.q_capacity
        self.soc_min = self.chemistry.lut_min_soc.interp(self.Tb_f)
        self.q_min = self.soc_min * self.q_capacity

        # Save and return
        # print('Mon CC:  charge_curr', charge_curr, 'dt', dt, 'd_delta_q', d_delta_q,'temp_lim', self.temp_lim, 'cap', self.q_capacity)
        return self.soc

    def load(self, delta_q):
        """Load states from retained memory"""
        self.delta_q = delta_q
