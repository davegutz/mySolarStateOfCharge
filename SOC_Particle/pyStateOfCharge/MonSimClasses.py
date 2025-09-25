# MonSimClasses:  Subclasses used to support replicate()
# Copyright (C) 2025 Dave Gutz
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

import numpy as np
import Battery
from Battery import Battery
from myFilters import LagExp
from pyDAGx import myTables

class TbSense:
    """Collect various sense parameters to create update delay in data feed"""

    def __init__(self, mon_ref=None, dTb_in=None):
        self.Tb0 = mon_ref.Tb_f[0]
        self.Tb0_s = mon_ref.Tb_mod[0]
        self.lut_dTb = None
        self.dTb = 0.
        if dTb_in is not None:
            self.dTb_in = np.array(dTb_in)
            self.Tb0 += dTb_in[1, 0]
            self.lut_dTb = myTables.TableInterp1D(np.array(dTb_in[0, :]), np.array(dTb_in[1, :]))
            self.dTb = lut_dTb.interp(t[0])
        self.Tb = mon_ref.Tb[0]
        self.Tb_f = mon_ref.Tb_f[0]
        self.Tb_f_rate = mon_ref.Tb_f_rate[0]
        self.Tb_past = mon_ref.Tb_rap[0] + self.dTb
        self.Tb_f_past = mon_ref.Tb_f_rap[0] + self.dTb
        self.Tb_f_rate_past = mon_ref.Tb_f_rate_rap[0]
        self.TbSenseFilt = LagExp(0, Battery.TB_FILT, Battery.TB_MIN, Battery.TB_MAX)

    def calc_dTb(self, i):
        if self.dTb is not 0.:
            dTb = ST.lut_dTb.interp(t[i])
        else:
            dTb = self.dTb
        return dTb

    def update(self):
        self.Tb_past = self.Tb
        self.Tb_f_past = self.Tb_f
        self.Tb_f_rate_past = self.Tb_f_rate

    def assign(self, mon_Tb, mon_Tb_f, mon_Tb_f_rate):
        self.Tb = mon_Tb + self.dTb
        self.Tb_f = mon_Tb_f + self.dTb
        self.Tb_f_rate = mon_Tb_f_rate
