#  installGUI_PlinkSOC  runs batches
#  .py
#  2026-Apr-26  Dave Gutz   Create
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
from test_soc_util import run_shell_cmd
from Colors import Colors
import shutil
import sys
import os
from pathlib import Path, PurePosixPath

# If shortcut disappears run debug = True to troubleshoot
debug = False

test_cmd_create = None
test_cmd_copy = None
result = None
GUI_PlinkSOC_dest_path = None

# Create executable
GUI_PlinkSOC_path = str(PurePosixPath(os.getcwd()) / 'GUI_PlinkSOC.png')
if sys.platform == 'win32':

    # Check executable is local
    if sys.executable.__contains__(str(Path("venv") / "Scripts" / "python")):
        pass
    else:
        print(Colors.fg.red, 'failed:  need to use local venv interpreter', Colors.reset)
        exit(1)

    GUI_PlinkSOC_dest_path = str(PurePosixPath(os.getcwd()) / 'dist' / 'GUI_PlinkSOC' / '_internal' / 'GUI_PlinkSOC.png')
    test_cmd_create = 'pyinstaller .\\GUI_PlinkSOC.py --i GUI_PlinkSOC.png -y'
    result = run_shell_cmd(test_cmd_create, silent=False)
    if result == -1:
        print(Colors.fg.red, 'failed', Colors.reset)
        exit(1)
    else:
        print(Colors.fg.green, 'success', Colors.reset)

    # Provide dependencies
    shutil.copyfile(GUI_PlinkSOC_path, GUI_PlinkSOC_dest_path)
    shutil.copystat(GUI_PlinkSOC_path, GUI_PlinkSOC_dest_path)
    print(Colors.fg.green, "copied files", Colors.reset)

# Install as deeply as possible
test_cmd_install = None
if sys.platform == 'linux':
    try:
        login = os.getlogin()
    except OSError:
        login = os.environ['LOGNAME']
    desktop_entry = f"""[Desktop Entry]
Name=GUI_PlinkSOC
Exec={sys.executable} /home/{login}/Documents/GitHub/mySolarStateOfCharge/SOC_Particle/pyStateOfCharge/GUI_PlinkSOC.py
Path=/home/{login}/Documents/GitHub/mySolarStateOfCharge/SOC_Particle/pyStateOfCharge
Icon=/home/{login}/Documents/GitHub/mySolarStateOfCharge/SOC_Particle/pyStateOfCharge/GUI_PlinkSOC.png
StartupWMClass=GUI_PlinkSOC
comment=app
Type=Application
Terminal=true
Encoding=UTF-8
Categories=Utility
"""

    with open(f"/home/{login}/Desktop/GUI_PlinkSOC.desktop", "w") as text_file:
        result = text_file.write("%s" % desktop_entry)
    if result == -1:
        print(Colors.fg.red, 'failed', Colors.reset)
    else:
        print(Colors.fg.green, 'success', Colors.reset)

    #  Launch permission
    test_cmd_launch = f'gio set /home/{login}/Desktop/GUI_PlinkSOC.desktop metadata::trusted true'
    result = run_shell_cmd(test_cmd_launch, silent=False)
    if result == -1:
        print(Colors.fg.red, 'gio set failed', Colors.reset)
    else:
        print(Colors.fg.green, 'gio set success', Colors.reset)
    test_cmd_perm = 'chmod a+x ~/Desktop/GUI_PlinkSOC.desktop'
    result = run_shell_cmd(test_cmd_perm, silent=False)
    if result == -1:
        print(Colors.fg.red, 'failed', Colors.reset)
    else:
        print(Colors.fg.green, 'success', Colors.reset)

    # Check executable is local
    if sys.executable.__contains__(str(Path("venv") / "bin" / "python")):
        pass
    else:
        print(Colors.fg.red, 'failed:  need to use local venv interpreter', Colors.reset)
        exit(1)

    # Execute permission
    test_cmd_perm = 'chmod a+x ~/Desktop/GUI_PlinkSOC.desktop'
    result = run_shell_cmd(test_cmd_perm, silent=False)
    if result == -1:
        print(Colors.fg.red, f"'chmod ...' failed code {result}", Colors.reset)
    else:
        print(Colors.fg.green, 'chmod success', Colors.reset)

    # Move file
    try:
        if debug:  # Leaves shortcut on desktop for troubleshooting
            pass
        else:
            result = shutil.move(f'/home/{login}/Desktop/GUI_PlinkSOC.desktop',
                                 '/usr/share/applications/GUI_PlinkSOC.desktop')
    except PermissionError:
        print(Colors.fg.red,
              "Stop and establish sudo permissions"
              "  or "
              f"sudo mv /home/{login}//Desktop/GUI_PlinkSOC.desktop /usr/share/applications/.",
              Colors.reset)
        exit(1)

    if result != '/usr/share/applications/GUI_PlinkSOC.desktop':
        if debug:
            print(Colors.fg.red, ".desktop file held on Desktop for debugging", Colors.reset)
        else:
            print(Colors.fg.red, f"'mv ...' failed code {result}", Colors.reset)
    else:
        print(Colors.fg.green,
              'mv success.  Browse apps :: and make it favorites.  Open and set path to dataReduction'
              "you shouldn't have to remake shortcuts",
              Colors.reset)

elif sys.platform == 'darwin':
    print(Colors.fg.green,
          f"Make sure 'Python Launcher' (Python Script Preferences) option for 'Allow override with #! in script' is checked.\n"
          f"in Finder double-click on 'GUI_PlinkSOC.png'.  Edit-copy the image\n"
          f"in Finder ctrl-click on 'GUI_PlinkSOC.py'\n"
          f"   - 'Get Info', click on 2nd icon, paste.   Drag item to taskbar",
          Colors.reset)
elif sys.platform == 'win32':
    print(Colors.fg.green,
          f"Browse to executable in 'dist/GUI_PlinkSOC' and double-click\n"
          f" Create shortcut first time and move Desktop\n"
          f" double-click on  'GUI_PlinkSOC.exe - Shortcut', set paths on buttons, pin to taskbar.")
    print(Colors.fg.red,
          f"In shortcut properties, make sure 'Start in:' is this folder where this script resides\n"
          f" After the first time you do this on a particular Windows install you shouldn't have to remake shortcuts\n",
          Colors.reset)
