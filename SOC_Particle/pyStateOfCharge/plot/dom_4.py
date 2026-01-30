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

""" dom_4 does..TODO
Dependencies:
    - numpy      (everything)
    - plq       GP plotter function
"""

import matplotlib.pyplot as plt
from plot.plq import plq as plq


def dom_4(mr, mv, sr, sv, smv, filename, plot_title=None, strict_overplot=False, fig_list=None, fig_files=None):
    fig_list.append(plt.figure())
    plt.subplot(131)
    plt.title(plot_title + ' DOM 4')
    print('DOM 4', end=':  ')
    plq(plt, mr, 'time', mr, 'soc', color='orange', linestyle='-')
    plq(plt, mv, 'time', mv, 'soc', color='green', linestyle='--')
    plq(plt, sr, 'time', sr, 'soc_s', color='red', linestyle='-.')
    plq(plt, smv, 'time', smv, 'soc_s', color='black', linestyle=':')
    plq(plt, mr, 'time', mr, 'soc_ekf', color='blue', linestyle='-')
    plq(plt, mv, 'time', mv, 'soc_ekf', color='cyan', linestyle='--')
    plt.legend(loc=1)
    plt.subplot(132)
    plq(plt, mr, 'time', mr, 'vb', color='orange', linestyle='-')
    plq(plt, mr, 'time', mr, 'vb_f', color='orange', linestyle='-', warn=False)
    plq(plt, mr, 'time', mr, 'vb_h', color='cyan', linestyle='--')
    plq(plt, mv, 'time', mv, 'vb', color='green', linestyle='-.')
    plq(plt, mr, 'time', mr, 'vb_s', color='red', linestyle='-.')
    plq(plt, smv, 'time', smv, 'vb_s', color='black', linestyle=':')
    plt.legend(loc=1)
    plt.subplot(133)
    plq(plt, mr, 'soc', mr, 'vb', color='orange', linestyle='-')
    plq(plt, mr, 'soc', mr, 'vb_f', color='orange', linestyle='-', warn=False)
    plq(plt, mr, 'soc', mr, 'vb_h', color='cyan', linestyle='--')
    plq(plt, mv, 'soc', mv, 'vb', color='green', linestyle='-.')
    plq(plt, mr, 'soc_s', mr, 'vb_s', color='red', linestyle='-.')
    plq(plt, smv, 'soc_s', smv, 'vb_s', color='black', linestyle=':')
    plt.legend(loc=1)
    fig_file_name = filename + '_' + str(len(fig_list)) + ".png"
    fig_files.append(fig_file_name)
    plt.savefig(fig_file_name, format="png")

    return fig_list, fig_files
