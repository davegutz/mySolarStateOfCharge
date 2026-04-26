# ComparePlotSettings.py:  used to rescale
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

import matplotlib.pyplot as plt
import textwrap
import sys
import os

if sys.platform == 'linux':
    import matplotlib
    is_wayland = os.environ.get('XDG_SESSION_TYPE') == 'wayland' or bool(os.environ.get('WAYLAND_DISPLAY'))
    if is_wayland:
        matplotlib.use('tkagg')
if sys.platform == 'darwin':
    import matplotlib
    matplotlib.use('tkagg')
plt.rcParams['axes.grid'] = True
plt.rcParams['legend.fontsize'] = 'small'
plt.rcParams['savefig.dpi'] = 500  # Set default savefig DPI to 300

# Monkey-patch savefig: resize to 16:9 for the save, restore display size after.
_orig_savefig = plt.savefig

def _savefig_fullscreen(*args, **kwargs):
    fig = plt.gcf()
    orig_size = fig.get_size_inches()
    fig.set_size_inches(16, 9)
    _orig_savefig(*args, **kwargs)
    fig.set_size_inches(*orig_size)

plt.savefig = _savefig_fullscreen

# Monkey-patch title/suptitle: wrap long strings so they don't overflow the figure edge.
_TITLE_WRAP_WIDTH = 70  # characters; fits comfortably in a 16-inch figure

_orig_title = plt.title
def _title_wrapped(label='', *args, **kwargs):
    if isinstance(label, str):
        label = textwrap.fill(label, width=_TITLE_WRAP_WIDTH)
    return _orig_title(label, *args, **kwargs)
plt.title = _title_wrapped

_orig_suptitle = plt.suptitle
def _suptitle_wrapped(t, *args, **kwargs):
    if isinstance(t, str):
        t = textwrap.fill(t, width=_TITLE_WRAP_WIDTH)
    return _orig_suptitle(t, *args, **kwargs)
plt.suptitle = _suptitle_wrapped


def rescale_time_axes(fig_list, t_min=None, t_max=None):
      """Rescale x-axes of all time subplots in every live figure without replotting.

      Applies to axes whose current x-range is NOT Unix epoch (i.e. 'time' and
      'time_t' axes, which are seconds relative to run start).
      Fault-plot axes using 'time_ux' (Unix epoch, values > 1e9) are skipped.

      Parameters
      ----------
      fig_list : list[matplotlib.figure.Figure]
          The curated list of live figures from compare_run_sim / dom_plot.
      t_min : float or None
          New left x-limit in seconds relative to run start.  None = keep current.
      t_max : float or None
          New right x-limit in seconds relative to run start.  None = keep current.

      Usage Examples

      # Zoom to minutes 5–30 of the run (300 s – 1800 s):
      rescale_time_axes(fig_list, t_min=300, t_max=1800)

      # Zoom in from the right only (keep left edge, cut off after 3600 s):
      rescale_time_axes(fig_list, t_max=3600)

      # Reset to autorange on all axes:
      for fig in fig_list:
          for ax in fig.axes:
              ax.autoscale(axis='x')
          fig.canvas.draw_idle()

      ---
      Key Design Notes

      ┌───────────────────────┬───────────────────────────────┬──────────────────┐
      │       Axis type       │           Detection           │   How handled    │
      ├───────────────────────┼───────────────────────────────┼──────────────────┤
      │ time / time_t         │ xlim < 1e9 (relative seconds) │ set_xlim applied │
      ├───────────────────────┼───────────────────────────────┼──────────────────┤
      │ time_ux (fault plots) │ xlim > 1e9 (Unix epoch)       │ skipped          │
      └───────────────────────┴───────────────────────────────┴──────────────────┘

      - fig.canvas.draw_idle() triggers a lazy redraw — the window updates on the next GUI event loop cycle, not instantly, which is the correct pattern for non-blocking interactive
      use with plt.show(block=False).
      - time_t axes (stair-step temperature subplots) share the same seconds-relative scale as time, so the same t_min/t_max applies to both correctly.

      """
      # noinspection PyPep8Naming
      UNIX_EPOCH_THRESHOLD = 1e9  # values this large are time_ux (Unix epoch), not relative time

      for fig in fig_list:
          changed = False
          for ax in fig.axes:
              xlo, xhi = ax.get_xlim()
              if xlo > UNIX_EPOCH_THRESHOLD or xhi > UNIX_EPOCH_THRESHOLD:
                  continue  # skip time_ux (fault-plot) axes
              new_lo = t_min if t_min is not None else xlo
              new_hi = t_max if t_max is not None else xhi
              if new_lo != xlo or new_hi != xhi:
                  ax.set_xlim(new_lo, new_hi)
                  changed = True
          if changed:
              fig.canvas.draw_idle()  # redraw without replotting
