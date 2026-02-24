# General purpose plot function including automatic labels and checks
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

""" General purpose plotter
Dependencies:
    - numpy      (everything)
"""

import numpy as np

def plq(plt_, sx, st, sy, yt, slr=1, add=0., color='black', linestyle='-', label=None, marker=None,
        markersize=None, markevery=None, stairs=False, warn=True, linewidth=1.5):
    if not label:
        slr_str = ''
        if slr != 1.:
            slr_str = str(slr) + '*'
        add_str = ''
        if add > 0:
            add_str = '+' + str(add)
        elif add < 0:
            add_str = str(add)
        if sy is not None and hasattr(sy, 'str') and sy.str is not None and sy.str!='':
            label = slr_str + yt + '_' + sy.str + add_str
        else:
            label = slr_str + yt + add_str
    if (sx is not None and sy is not None and hasattr(sx, st) and hasattr(sy, yt) and getattr(sy, yt) is not None and
            len(getattr(sy, yt)) > 0 and getattr(sy, yt)[0] is not None):
        try:
            yscld = getattr(sy, yt) * slr + add
        except TypeError:
            yscld = np.array(getattr(sy, yt)) * slr + add
        try:
            if stairs:
                try:
                    dt = getattr(sx, st)[-1] - getattr(sx, st)[-2]
                except IndexError:
                    if warn:
                        print(f"plq: skipping     {yt}({st})     labeled  '{label}'  Dimensions of time different")
                    return
                x_in = np.append(getattr(sx, st), getattr(sx, st)[-1]+dt)
                plt_.stairs(yscld, x_in, color=color, linestyle=linestyle, label=label, baseline=None,
                            linewidth=linewidth)
            else:
                plt_.plot(getattr(sx, st), yscld, color=color, linestyle=linestyle, label=label, marker=marker,
                          markersize=markersize, markevery=markevery, linewidth=linewidth)
        except ValueError:
            if warn:
                print(f"plq: skipping     {yt}({st})     labeled  '{label}'")
    else:
        if warn:
            print(f"plq: skipping     {yt}({st})     labeled  '{label}'")

