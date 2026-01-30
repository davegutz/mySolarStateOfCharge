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

""" dom_4a does..TODO
Dependencies:
    - numpy      (everything)
    - plq       GP plotter function
"""

import matplotlib.pyplot as plt
from plot.plq import plq as plq


def dom_4a(mr, mv, sr, sv, smv, filename, plot_title=None, strict_overplot=False, fig_list=None, fig_files=None):
    fig_list.append(plt.figure())
    plt.subplot(311)
    plt.title(plot_title + ' DOM 4a')
    print('DOM 4a', end=':  ')
    plq(plt, mr, 'time', mr, 'ib', color='orange', linestyle='-')
    plq(plt, mr, 'time', mr, 'ib_f', color='orange', linestyle='-', warn=False)
    plq(plt, mv, 'time', mv, 'ib', color='green', linestyle='--')
    plq(plt, mr, 'time', mr, 'ib_charge', color='red', linestyle='-.')
    plq(plt, mr, 'time', mr, 'ib_charge_f', color='red', linestyle='-.', warn=False)
    plq(plt, mv, 'time', mv, 'ib_charge', color='black', linestyle=':')
    plt.legend(loc=1)
    plt.subplot(312)
    plq(plt, sr, 'time', sr, 'soc_s', color='orange', linestyle='-')
    plq(plt, smv, 'time', smv, 'soc_s', color='green', linestyle='--')
    plq(plt, mr, 'time', mr, 'soc', color='red', linestyle='-.')
    plq(plt, mv, 'time', mv, 'soc', color='black', linestyle=':')
    plt.legend(loc=1)
    plt.subplot(313)
    plq(plt, mr, 'time', mr, 'Tb_rap', color='red', linestyle='-')
    plq(plt, mv, 'time', mv, 'Tb_rap', color='cyan', linestyle='--')
    plt.legend(loc=1)
    fig_file_name = filename + '_' + str(len(fig_list)) + ".png"
    fig_files.append(fig_file_name)
    plt.savefig(fig_file_name, format="png")

    return fig_list, fig_files
