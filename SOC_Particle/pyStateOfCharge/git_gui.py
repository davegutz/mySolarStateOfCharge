#! /bin/sh
"exec" "`dirname $0`/.venv/bin/python3" "$0" "$@"
# git_gui.py
# 2026-Apr-02  Dave Gutz  Create
# Copyright (C) 2026 Dave Gutz
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation;
# version 2.1 of the License.
#
# Simple Tkinter GUI for running common git commands.
# Always operates in the directory from which this script was launched.

import sys
import os
import subprocess
import threading
import tkinter as tk
from tkinter import scrolledtext, simpledialog, messagebox

# Set terminal window title
sys.stdout.write("\033]0;Git GUI\007")
sys.stdout.flush()

# Capture the working directory at launch time
REPO_DIR = os.getcwd()

# ---------------------------------------------------------------------------
# Button definitions: (label, command_list)
# None for command_list means the button has custom logic (e.g. needs input)
# ---------------------------------------------------------------------------
INSPECT_BUTTONS = [
    ("Status",     ["git", "status"]),
    ("Log",        ["git", "log", "--oneline", "-25"]),
    ("Diff",       ["git", "diff"]),
    ("Diff Staged",["git", "diff", "--staged"]),
    ("Branches",   ["git", "branch", "-a"]),
    ("Remotes",    ["git", "remote", "-v"]),
]

ACTION_BUTTONS = [
    ("Add All",    ["git", "add", "--all"]),
    ("Commit...",  None),   # prompts for message
    ("Push",       ["git", "push"]),
    ("Fetch",      ["git", "fetch"]),
    ("Pull",       ["git", "pull"]),
    ("Stash",      ["git", "stash"]),
    ("Stash Pop",  ["git", "stash", "pop"]),
    ("Stash List", ["git", "stash", "list"]),
]

BG_COLOR  = '#1e1e1e'
FG_COLOR  = '#d4d4d4'
CMD_COLOR = '#4ec9b0'   # teal  — echoed command line
ERR_COLOR = '#f48771'   # salmon — stderr / non-zero exit
OK_COLOR  = '#b5cea8'   # soft green — exit 0 notice
SEP_COLOR = '#569cd6'   # blue — separator


class GitGui:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(f"Git GUI  —  {REPO_DIR}")
        self.root.configure(bg=BG_COLOR)
        self._build_buttons()
        self._build_output()
        self._build_statusbar()
        self.log(f"Repo: {REPO_DIR}\n", tag='sep')

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------
    def _build_buttons(self):
        outer = tk.Frame(self.root, bg=BG_COLOR)
        outer.pack(fill=tk.X, padx=6, pady=(6, 2))

        # Row labels
        tk.Label(outer, text="Inspect", bg=BG_COLOR, fg=SEP_COLOR,
                 font=('Courier', 11, 'bold')).grid(row=0, column=0, sticky='w', padx=(0, 6))
        tk.Label(outer, text="Actions", bg=BG_COLOR, fg=SEP_COLOR,
                 font=('Courier', 11, 'bold')).grid(row=1, column=0, sticky='w', padx=(0, 6))

        for col, (label, cmd) in enumerate(INSPECT_BUTTONS, start=1):
            tk.Button(outer, text=label, width=11, relief=tk.FLAT,
                      bg='#3c3c3c', fg=FG_COLOR, activebackground='#505050',
                      font=('Courier', 11),
                      command=lambda c=cmd, l=label: self._dispatch(c, l)
                      ).grid(row=0, column=col, padx=2, pady=2)

        for col, (label, cmd) in enumerate(ACTION_BUTTONS, start=1):
            tk.Button(outer, text=label, width=11, relief=tk.FLAT,
                      bg='#264f78', fg=FG_COLOR, activebackground='#3a6ea5',
                      font=('Courier', 11),
                      command=lambda c=cmd, l=label: self._dispatch(c, l)
                      ).grid(row=1, column=col, padx=2, pady=2)

        # Clear button far right
        max_col = max(len(INSPECT_BUTTONS), len(ACTION_BUTTONS)) + 2
        tk.Button(outer, text="Clear", width=7, relief=tk.FLAT,
                  bg='#5a1e1e', fg=FG_COLOR, activebackground='#7a2e2e',
                  font=('Courier', 11),
                  command=self._clear
                  ).grid(row=0, column=max_col, rowspan=2, padx=(8, 2), pady=2, sticky='ns')

    def _build_output(self):
        self.output = scrolledtext.ScrolledText(
            self.root, wrap=tk.WORD, height=32, width=100,
            bg=BG_COLOR, fg=FG_COLOR, insertbackground=FG_COLOR,
            font=('Courier', 12), state=tk.DISABLED,
            relief=tk.FLAT, borderwidth=0)
        self.output.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        self.output.tag_config('cmd', foreground=CMD_COLOR, font=('Courier', 12, 'bold'))
        self.output.tag_config('err', foreground=ERR_COLOR)
        self.output.tag_config('ok',  foreground=OK_COLOR)
        self.output.tag_config('sep', foreground=SEP_COLOR)

    def _build_statusbar(self):
        self.status_var = tk.StringVar(value=f"Ready  |  {REPO_DIR}")
        bar = tk.Label(self.root, textvariable=self.status_var, anchor='w',
                       bg='#007acc', fg='white', font=('Courier', 11),
                       padx=6, pady=2)
        bar.pack(fill=tk.X, side=tk.BOTTOM)

    # ------------------------------------------------------------------
    # Output helpers (always called from main thread via root.after)
    # ------------------------------------------------------------------
    def log(self, text: str, tag: str = None):
        self.output.configure(state=tk.NORMAL)
        self.output.insert(tk.END, text, tag or '')
        self.output.see(tk.END)
        self.output.configure(state=tk.DISABLED)

    def _clear(self):
        self.output.configure(state=tk.NORMAL)
        self.output.delete('1.0', tk.END)
        self.output.configure(state=tk.DISABLED)
        self.status_var.set(f"Cleared  |  {REPO_DIR}")

    # ------------------------------------------------------------------
    # Command dispatch
    # ------------------------------------------------------------------
    def _dispatch(self, cmd, label):
        if cmd is None:
            # Commit — prompt for message
            msg = simpledialog.askstring("Commit Message", "Enter commit message:",
                                         parent=self.root)
            if not msg or not msg.strip():
                self.log("Commit cancelled.\n", tag='sep')
                return
            cmd = ["git", "commit", "-m", msg.strip()]

        self.log(f"\n$ {' '.join(cmd)}\n", tag='cmd')
        self.status_var.set(f"Running: {' '.join(cmd)}")
        self.root.update_idletasks()

        threading.Thread(target=self._run, args=(cmd,), daemon=True).start()

    def _run(self, cmd):
        try:
            result = subprocess.run(
                cmd, cwd=REPO_DIR,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout = result.stdout
            stderr = result.stderr
            rc     = result.returncode
        except FileNotFoundError as exc:
            self.root.after(0, lambda: self.log(f"Error: {exc}\n", tag='err'))
            self.root.after(0, lambda: self.status_var.set("Error"))
            return

        def _update():
            if stdout:
                self.log(stdout)
            if stderr:
                self.log(stderr, tag='err')
            if rc == 0:
                self.log(f"[exit 0]\n", tag='ok')
            else:
                self.log(f"[exit {rc}]\n", tag='err')
            self.status_var.set(
                f"Done: {' '.join(cmd)}  [exit {rc}]  |  {REPO_DIR}")

        self.root.after(0, _update)


# ---------------------------------------------------------------------------
def main():
    root = tk.Tk()
    root.geometry("1000x640")
    GitGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
