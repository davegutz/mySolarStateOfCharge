# PlotKiller: class called after plt.show to block and take input from user to then close all plots.  Viewable
# on taskbar.
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

"""
Raise a window visible at task bar to close all plots.
  *** Call ion before show in caller, or use 'show_and_kill()' instead of 'plt.show(); show_killer();
  Sometimes works without ion.  (Race condition in IPC)
  https://stackoverflow.com/questions/28269157/plotting-in-a-non-blocking-way-with-matplotlib
  *** sometimes need to send list of figures to prevent close('all') from closing all.  (Again, race condition in IPC)
"""

import matplotlib.pyplot as plt
import time
import platform
if platform.system() == 'Darwin':
    import tkinter as tk
    # noinspection PyUnresolvedReferences
    from ttwidgets import TTButton as myButton
else:
    import tkinter as tk
    # from tkinter import Button as myButton
bg_color = "lightgray"
from ComparePlotSettings import rescale_time_axes


def do_hardcopy(fig_list, fig_files, pdf_path, pdf_base):
    """Save figures to PNGs on the main thread, then assemble the PDF in a daemon thread."""
    from datetime import datetime
    from unite_pictures import precleanup_fig_files, pngs_to_pdf
    import threading
    if not fig_list or not fig_files or not pdf_base:
        return
    try:
        for fig, fig_file in zip(fig_list, fig_files):
            plt.figure(fig.number)
            plt.savefig(fig_file, format="png")
            print("saved", fig_file)
        date_time = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        def _assemble(base=pdf_base, path=pdf_path, dt=date_time):
            try:
                precleanup_fig_files(output_pdf_name=base, path_to_pdfs=path)
                print('\ncreating pdf...')
                pngs_to_pdf(png_folder=path, output_pdf=base + '_' + dt + '.pdf')
            except Exception as e:
                print(f"do_hardcopy pdf ERROR: {e}")
        threading.Thread(target=_assemble, daemon=True).start()
    except Exception as e:
        print(f"do_hardcopy ERROR: {e}")


class PlotKiller(tk.Toplevel):
    # noinspection PyUnusedLocal
    def __init__(self, message, caller, fig_list_=None, fig_files_=None, pdf_path_='.', pdf_base_=None):
        """Block caller task asking to close all plots then doing so"""
        self.fig_list = fig_list_
        self.fig_files = fig_files_
        self.pdf_path = pdf_path_
        self.pdf_base = pdf_base_
        tk.Toplevel.__init__(self)
        self.title("SOC-close")
        tk.Button(self, command=self.close_figs, text="close " + message, font=("Courier", 12)).grid(row=0, column=0, columnspan=4, padx=15, pady=15)
        tk.Label(self, text="t_min:", font=("Courier", 10)).grid(row=1, column=0, padx=5, pady=5, sticky='e')
        self.t_min_var = tk.StringVar()
        tk.Entry(self, textvariable=self.t_min_var, width=10, font=("Courier", 10)).grid(row=1, column=1, padx=5, pady=5)
        tk.Label(self, text="t_max:", font=("Courier", 10)).grid(row=1, column=2, padx=5, pady=5, sticky='e')
        self.t_max_var = tk.StringVar()
        tk.Entry(self, textvariable=self.t_max_var, width=10, font=("Courier", 10)).grid(row=1, column=3, padx=5, pady=5)
        tk.Button(self, command=self.rescale_axes, text="rescale", font=("Courier", 10)).grid(row=2, column=0, columnspan=4, padx=15, pady=5)
        if fig_files_ is not None and pdf_base_ is not None:
            tk.Button(self, command=self.hardcopy, text="Hardcopy", font=("Courier", 10)).grid(row=3, column=0, columnspan=4, padx=15, pady=5)
        self.lift()
        self.mainloop()
        # self.grab_set()  # Prevents other Tkinter windows from being used

    def rescale_axes(self):
        t_min_str = self.t_min_var.get().strip()
        t_max_str = self.t_max_var.get().strip()
        t_min = float(t_min_str) if t_min_str else None
        t_max = float(t_max_str) if t_max_str else None
        if self.fig_list is not None:
            rescale_time_axes(self.fig_list, t_min=t_min, t_max=t_max)

    def hardcopy(self):
        do_hardcopy(self.fig_list, self.fig_files, self.pdf_path, self.pdf_base)

    def close_figs(self):
        if self.fig_list is None:
            plt.close('all')
        else:
            for fig in self.fig_list:
                plt.close(fig)
        # self.grab_release()
        self.destroy()


def show_and_kill(string, caller, fig_list=None, fig_files=None, pdf_path='.', pdf_base=None, hardcopy=False):
    plt.show()
    time.sleep(1)
    if hardcopy:
        do_hardcopy(fig_list, fig_files, pdf_path, pdf_base)
    PlotKiller(string, caller, fig_list, fig_files, pdf_path, pdf_base)


def show_killer(string, caller, fig_list=None, fig_files=None, pdf_path='.', pdf_base=None, hardcopy=False):
    if hardcopy:
        do_hardcopy(fig_list, fig_files, pdf_path, pdf_base)
    PlotKiller(string, caller, fig_list, fig_files, pdf_path, pdf_base)


def simple_plot1():
    import numpy as np
    fig_list = []
    t = np.arange(0.0, 2.0, 0.01)
    s = 1 + np.sin(2 * np.pi * t)
    fig, ax = plt.subplots()
    fig_list.append(fig)
    ax.plot(t, s)
    ax.set(xlabel='time (s)', ylabel='voltage (mV)',
           title='Sine wave1')
    ax.grid()
    fig, ax = plt.subplots()
    fig_list.append(fig)
    ax.plot(t, s)
    ax.set(xlabel='time (s)', ylabel='voltage (mV)',
           title='Sine wave2')
    ax.grid()
    plt.ion()
    plt.show()
    show_killer('close plots?', 'sp1', fig_list)


def simple_plot2():
    import numpy as np
    t = np.arange(0.0, 2.0, 0.01)
    s = 1 + np.sin(2 * np.pi * t)
    fig, ax = plt.subplots()
    ax.plot(t, s)
    ax.set(xlabel='time (s)', ylabel='voltage (mV)',
           title='Sine wave1')
    ax.grid()
    fig, ax = plt.subplots()
    ax.plot(t, s)
    ax.set(xlabel='time (s)', ylabel='voltage (mV)',
           title='Sine wave2')
    ax.grid()
    plt.ion()
    # plt.show()
    show_and_kill('close plots?', 'sp2')


if __name__ == '__main__':  # Example usage.  Ran ok 20260217
    root = tk.Tk()
    tk.Label(root, text="Try opening multiple plots then killing").pack()
    tk.Button(root, text="plot 1", command=simple_plot1).pack()
    tk.Button(root, text="plot 2", command=simple_plot2).pack()
    root.mainloop()
