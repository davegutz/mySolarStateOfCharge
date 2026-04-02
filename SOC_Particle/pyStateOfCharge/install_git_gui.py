#! /usr/bin/env python3
# install_git_gui.py
# 2026-Apr-02  Dave Gutz  Create
# Copyright (C) 2026 Dave Gutz
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation;
# version 2.1 of the License.
#
# Installs the 'gg' command to ~/.local/bin so that running 'gg' from
# any directory launches git_gui.py in that folder.

import os
import stat
import sys

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
GIT_GUI_PY  = os.path.join(SCRIPT_DIR, 'git_gui.py')
VENV_PYTHON = os.path.join(SCRIPT_DIR, '.venv', 'bin', 'python3')
BIN_DIR     = os.path.expanduser('~/.local/bin')
GG_PATH     = os.path.join(BIN_DIR, 'gg')


def main():
    if not os.path.isfile(GIT_GUI_PY):
        print(f"ERROR: {GIT_GUI_PY} not found.", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(VENV_PYTHON):
        print(f"ERROR: venv python not found at {VENV_PYTHON}", file=sys.stderr)
        print("Set up the venv first:", file=sys.stderr)
        print(f"  python3 -m venv {os.path.join(SCRIPT_DIR, '.venv')}", file=sys.stderr)
        print(f"  {VENV_PYTHON} -m pip install -r {os.path.join(SCRIPT_DIR, 'requirements.txt')}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(BIN_DIR, exist_ok=True)

    wrapper = f"""#!/bin/sh
# gg — launch git_gui.py in the current working directory
exec "{VENV_PYTHON}" "{GIT_GUI_PY}" "$@"
"""
    with open(GG_PATH, 'w') as f:
        f.write(wrapper)

    current_mode = os.stat(GG_PATH).st_mode
    os.chmod(GG_PATH, current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"Installed: {GG_PATH}")

    path_dirs = os.environ.get('PATH', '').split(':')
    if BIN_DIR not in path_dirs:
        print(f"\nNOTE: {BIN_DIR} is not yet on your PATH.")
        print("Add this to your ~/.bashrc (or ~/.zshrc), then open a new terminal:")
        print(f'  export PATH="$HOME/.local/bin:$PATH"')
    else:
        print("Ready — run 'gg' from any directory to open Git GUI there.")


if __name__ == '__main__':
    main()
