# Installing Development Tools for SOC_Particle

This guide covers everything needed to build, flash, and run the SOC_Particle firmware and its Python companion tools on **Ubuntu/Lubuntu/Pop!_OS** and **Windows**.

For macOS see [doc/InstallationMacOS.md](doc/InstallationMacOS.md).

---

## Table of Contents

- [1. Git and GitHub Desktop](#1-git-and-github-desktop)
- [2. Clone the Repository](#2-clone-the-repository)
- [3. VS Code and Particle Workbench](#3-vs-code-and-particle-workbench)
- [4. PyCharm and Python Environment](#4-pycharm-and-python-environment)
- [5. puTTY Serial Terminal](#5-putty-serial-terminal)
- [6. First Flash to Photon2](#6-first-flash-to-photon2)
- [7. Running the Python GUI](#7-running-the-python-gui)

---

## 1. Git and GitHub Desktop

### Ubuntu / Lubuntu

```bash
sudo apt install -y git
sudo wget https://github.com/shiftkey/desktop/releases/download/release-3.1.1-linux1/GitHubDesktop-linux-3.1.1-linux1.deb
sudo apt-get install -y gdebi-core
sudo gdebi GitHubDesktop-linux-3.1.1-linux1.deb
```

When GitHub Desktop first opens, sign in to GitHub.com when prompted.
**Do not** log in proactively via the browser first — wait for the app to request it to avoid an authorization loop.

### Pop!_OS

Same as Ubuntu above.

### Windows

- Download and run the GitHub Desktop installer from https://desktop.github.com/
- Sign in to GitHub.com via the browser first, then open GitHub Desktop and clone.

---

## 2. Clone the Repository

In GitHub Desktop: **File → Clone repository → URL**

```
https://github.com/davegutz/mySolarStateOfCharge
```

Clone into a location such as `Documents/GitHub/`.

To clone from the command line:

```bash
git clone https://github.com/davegutz/mySolarStateOfCharge
```

---

## 3. VS Code and Particle Workbench

### Ubuntu / Lubuntu

> **Use the `.deb` package — do NOT install via `snap install code --classic`.**
> The snap version sandboxes VS Code and blocks the Particle build tools from running.

```bash
# Download the .deb package from https://code.visualstudio.com/docs/?dv=linux64_deb
sudo apt install ./Downloads/code_*.deb
# Ignore "performed unsandboxed as root" warnings — they are harmless
```

Fix the missing `crc32` tool that the Particle build requires:

```bash
sudo apt install libarchive-zip-perl
```

### Pop!_OS (COSMIC)

> **Do NOT install via the Cosmic Store / flatpak — that also sandboxes VS Code.**

Same `.deb` install as Ubuntu above.

### Windows

Download and run the VS Code installer from https://code.visualstudio.com/

### All platforms — Particle Workbench extension

1. Open VS Code.
2. Go to **Extensions** (Ctrl+Shift+X).
3. Search for **Particle Workbench** and install it.
4. **Sign in to Particle immediately** (click the Particle icon → Log In):
   - Username: `davegutz@alum.mit.edu`
   - The first sign-in is needed before any build or flash will work.

Recommended additional extensions.  Really none are needed but I played with these.
- **Codeium AI** (log in via Google)
- **Python** (Microsoft)

### Opening the project

- **File → Open Folder** → browse to `Documents/GitHub/myStateOfCharge/SOC_Particle`
- In the Explorer panel: **File Icon → Explorer → Open Editors**
- Use the search icon with the filter `*.h, *.cpp, *.ino` and exclude `SOC_Particle.cpp`

### Selecting the target device

Open `src/local_config.h` and uncomment the line for your hardware unit, e.g.:

```cpp
#include "soc3p2_hi_lo.h"   // guest room system
// #include "soc4p2_hi_lo.h" // truck system
```

Then in VS Code: **Ctrl+Shift+P → Particle: Configure for Device** → choose the OS version and device name.

---

## 4. PyCharm and Python Environment

The Python companion tool (`pyStateOfCharge/GUI_TestSOC.py`) runs in PyCharm.

### Ubuntu / Lubuntu

```bash
sudo apt install -y python3-tk          # required for Tkinter GUI
sudo snap install pycharm-community --classic
```

> **Caution:** Use PyCharm (not the command line) to install Python interpreters.
> Installing Python via the Debian package manager can break the system Python.

### Pop!_OS (COSMIC)

Pop!_OS uses Flatpak by default, which prevents PyCharm from finding custom Python builds.
Install via `snap` instead:

```bash
sudo apt install -y python3-tk
sudo snap install snapd
# log out and log back in after installing snapd
sudo snap install pycharm-community --classic
```

### Windows

- Download PyCharm Community from https://www.jetbrains.com/pycharm/download/
- Run the installer; quit the JetBrains Toolbox from the system tray after installation.
- Install Python 3.12 from https://www.python.org/ — choose **non-admin install** and tick **Add Python to PATH**.

### Setting up the Python virtual environment

1. Open PyCharm and open the folder `SOC_Particle/pyStateOfCharge/`.
2. Create a new virtual environment (venv) using the local Python interpreter.
3. Let PyCharm index the project (this takes a minute on first open).
4. Run `install.py` from PyCharm to install all required packages.
   - If `pyinstaller` is missing, install it in the interpreter before running `install.py`.

---

## 5. puTTY Serial Terminal

puTTY is required for the `Talk` interface to the Photon2 and integrates with `GUI_TestSOC.py`.

### Ubuntu / Lubuntu

```bash
sudo apt-get install -y putty
```

Grant your user permission to access the USB serial device without `sudo`:

**Ubuntu/Lubuntu:**
```bash
sudo usermod -aG dialout $USER
# log out and log back in
```

**Pop!_OS:**
```bash
sudo adduser $USER dialout
# log out and log back in
```

Verify the device appears after plugging in the Photon2:
```bash
sudo dmesg | grep tty   # find the device, usually /dev/ttyACM0
```

### Windows

- Download the puTTY installer from https://www.putty.org/
- Find the COM port: **Device Manager → Ports (COM & LPT) → USB Serial Device (COM#)**
- Add the puTTY folder to the system `PATH` environment variable so `GUI_TestSOC.py` can call it.

### Configuring puTTY sessions

puTTY needs two saved sessions: `def` (default/idle) and `test` (active data capture).

Follow the step-by-step screenshots in:
- [dataReduction/putty/puTTY_Windows_setup_def.odt](dataReduction/putty/puTTY_Windows_setup_def.odt) — the `def` session.  Start with this then add extra stuff from test.odt for 'test' session.
- [dataReduction/putty/puTTY_Windows_setup_test.odt](dataReduction/putty/puTTY_Windows_setup_test.odt) — the `test` session

Key settings for both sessions:
- **Connection type:** Serial
- **Serial line:** the device port (e.g. `COM3` on Windows, `/dev/ttyACM0` on Linux)
- **Speed:** 230400
- **Font:** Free Mono 10 (Linux) or Courier New 10 (Windows)
- **Logging:** set the log file path to your data reduction folder (not Google Drive — use local storage)

After configuring `def`, load it and clone it as `test`, then change only the logging path in `test`.

Quick check — the Photon2 will not always produce output on connection.
Type `vv1;` in the puTTY terminal to start a data stream.
Type `vv0;` to stop it.
Type `h;` for help.

---

## 6. First Flash to Photon2

The first time you flash a new Particle device, the Device OS must also be flashed:

1. Plug the Photon2 into USB.
2. In VS Code: **Ctrl+Shift+P → Particle: Flash Application and Device OS (local)**
3. When prompted to install USB drivers, answer **Y** and enter your password.
   Ignore "missing permissions" messages — they clear on the next attempt.
4. Subsequent flashes only need **Particle: Flash Application (local)**.

If the device is not found:
- Try a different USB port (not a hub).
- Try a different cable (data cable, not charge-only).
- Put the device in DFU mode: hold MODE + RESET until blinking yellow, then release RESET.

If the build fails with `STM32_Pin_Info does not name a type`:
- The wrong config is `#include`d in `local_config.h`. Select the correct unit file.

If the device shows SOS 4 (bus fault) after flash:
- Reduce `NSUM` in `constants.h` — you have exceeded SRAM.

---

## 7. Running the Python GUI

`GUI_TestSOC.py` automates puTTY sessions and data collection.

1. Open PyCharm and open `SOC_Particle/pyStateOfCharge/GUI_TestSOC.py`.
2. Run it (the venv must be set up per [Section 4](#4-pycharm-and-python-environment)).
3. Set the **Data Reduction Folder** to your local data folder (not Google Drive for active sessions).
4. Work top-to-bottom through the GUI buttons:
   - **Init** — starts puTTY, copies the init string to the clipboard. Paste it into puTTY (right-click on Windows; Ctrl+Shift+V on Linux).
   - Handle any device prompts (or reset the device if you pasted into a dialog by accident).
   - **Start** — paste the start string into puTTY, then click **Reset** to start the timer.
   - Wait for the asterisks (`*`) to stop — the device is initializing.
   - **Done** — end the run and save.
   - **Compare** — overplot the run against the simulation model.

> Note: SRAM is written every ~10 seconds. If you reset or depower before then,
> any `*` adjustments made during that window will be lost.
