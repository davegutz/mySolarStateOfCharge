# PlotOptions - argument list consolidation
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

from dataclasses import dataclass
from typing import Optional
from SavedData import SavedData, SavedDataSim


@dataclass
class PlotOptions:
    mr: SavedData
    mv: Optional[SavedData]
    sr: Optional[SavedDataSim] = None
    sv: Optional[SavedDataSim] = None
    smr: Optional[SavedDataSim] = None
    smv: Optional[SavedDataSim] = None
    filename: Optional[str] = ''
    plot_title: Optional[str] = ''
    strict_overplot: Optional[bool] = False
    run_type: Optional[str] = ''

    def __init__(self, mr=None, mv=None, sr=None, sv=None, smr=None, smv=None, filename=None, plot_title=None,
                 strict_overplot=None, run_type='None'):
        self.mr = mr
        self.mv = mv
        self.sr = sr
        self.sv = sv
        self.smr = smr
        self.smv = smv
        self.run_type = run_type
        self.run_is_trans = self.run_type=='RunSim' or self.run_type=='RunRun'
        self.ver_is_trans_sim = self.run_type == 'RunSim'
        self.ver_is_trans_run = self.run_type == 'RunRun'
        self.run_is_stdy = not self.run_is_trans
        self.ver_is_stdy = not self.ver_is_trans_run and not self.ver_is_trans_sim

