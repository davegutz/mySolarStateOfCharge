# SOC_Particle — Quick Start Guide

End-to-end setup for the SOC_Particle development environment on **Linux**,
**Windows**, and **macOS**, in that order. The goal is a working installation
that can clone the repository, build and flash firmware to a Particle Photon 2,
talk to it over USB serial, and run the Python regression GUI.

For deeper installation notes and platform-specific quirks see
[INSTALL.md](INSTALL.md), [doc/InstallationLinux.md](doc/InstallationLinux.md),
[doc/InstallationWindows.md](doc/InstallationWindows.md), and
[doc/InstallationMacOS.md](doc/InstallationMacOS.md). For background on the
project see [README.md](README.md) and the
[Technical White Paper](README_WhitePaper.md).

---

## Table of Contents

- [Common Prerequisites](#common-prerequisites)
- [Linux](#linux)
- [Windows 11](#windows-11)
- [macOS](#macos)
- [After Setup — Sanity Check](#after-setup--sanity-check)
- [Troubleshooting Quick Reference](#troubleshooting-quick-reference)

---

## Common Prerequisites

All three platforms need the same five things, installed in roughly the same
order:

1. **Git / GitHub Desktop** — to clone the repository.
2. **VS Code with the Particle Workbench extension** — to build and flash
   firmware to the Photon 2.
3. **PyCharm Community + Python 3.12** — to run the regression and
   data-reduction GUI (`pyStateOfCharge/GUI_TestSOC.py`).
4. **puTTY** — for the `Talk` serial interface to the device.
5. **The Photon 2 hardware** — see
   [Off-the-Shelf Hardware Description](README.md#off-the-shelf-hardware-description)
   in the main README.

Sign in to a Particle account at <https://www.particle.io/> *before* installing
Particle Workbench — the first build will not run until you have signed in.

---

## Linux

Tested on Ubuntu, Lubuntu 24.04 (Lenovo), and Pop!_OS 24.04 (COSMIC). Most
steps are identical across Ubuntu derivatives.

### 1. Basic system packages (Linux)

```bash
sudo apt update
sudo apt upgrade
sudo apt install -y git curl wget gdebi-core
sudo apt install -y fuse libfuse2          # AppImage support
sudo apt install -y python3-pip python3-tk # python3-tk required for PyCharm GUIs
sudo apt install -y libarchive-zip-perl    # provides crc32 — Particle Workbench needs it
sudo apt install -y dos2unix xsel
```

If you are on **Lubuntu** with only 4 GB of RAM, also enlarge swap to ≥ 8 GB
and lower swappiness — PyCharm indexing and Particle builds will thrash
otherwise.

Lower swappiness from the default 60 to 10:

```bash
cat /proc/sys/vm/swappiness          # confirm current value
sudo sysctl vm.swappiness=10         # apply for this session
sudo nano /etc/sysctl.conf           # persist across reboots — add line:
# vm.swappiness=10
```

Enlarge swap to 8 GB:

```bash
swapon --show
sudo swapoff /swapfile
sudo fallocate -l 8G /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
swapon --show                        # confirm new size
```

### 2. GitHub Desktop and clone

```bash
cd ~/Downloads
wget https://github.com/shiftkey/desktop/releases/download/release-3.1.1-linux1/GitHubDesktop-linux-3.1.1-linux1.deb
sudo gdebi GitHubDesktop-linux-3.1.1-linux1.deb
github
```

When GitHub Desktop launches, **let it prompt you for the GitHub sign-in** —
do not log in to GitHub in the browser first; that triggers an authorization
loop. Sign in when prompted, then **File → Clone repository → URL**:

```text
https://github.com/davegutz/mySolarStateOfCharge
```

Clone into `~/Documents/GitHub/`.

### 3. VS Code and Particle Workbench (Linux)

> **Use the official `.deb` package — NOT `snap install code --classic` and
> NOT the Pop!_OS Cosmic Store flatpak.** Both sandbox VS Code in a way that
> blocks the Particle build toolchain.

```bash
# Download .deb from https://code.visualstudio.com/docs/?dv=linux64_deb
sudo apt install ~/Downloads/code_*.deb
```

Install the **Particle Workbench** extension:

1. Launch VS Code → Extensions panel (Ctrl+Shift+X).
2. Search **Particle Workbench** → Install.
3. Click the Particle icon in the activity bar → **Log In** with your Particle
   account.

### 4. PyCharm and Python venv

```bash
sudo snap install pycharm-community --classic
```

> Pop!_OS specific: install `snapd` first (`sudo apt install snapd`), then
> **log out and back in** before `snap install`.

Use PyCharm itself to manage Python interpreters — *do not* manually replace
the system Python.

In PyCharm:

1. **Open** → browse to `Documents/GitHub/mySolarStateOfCharge/SOC_Particle/pyStateOfCharge/`.
2. Settings → Project → Python Interpreter → Add → New Virtualenv (use the
   system `python3` as the base).
3. Let PyCharm finish indexing.
4. Right-click `install.py` → Run. This pulls in every required Python
   package.

### 5. puTTY and serial access

```bash
sudo apt install -y putty
sudo usermod -aG dialout $USER
```

Log out and log back in for the group change to take effect. Then verify the
Photon 2 appears when plugged in:

```bash
sudo dmesg | grep tty   # expect /dev/ttyACM0
```

Configure two saved puTTY sessions named `def` and `test` per the screenshots
in [dataReduction/putty/](dataReduction/putty/). Key settings for both:

- Connection type: **Serial**
- Serial line: `/dev/ttyACM0`
- Speed: **230400**
- Font: Free Mono 10
- Logging: local folder (not Google Drive)

### 6. First flash (Linux)

1. Plug the Photon 2 into a USB port (use a data cable; not a hub if possible).
2. In VS Code with the project open, open `src/local_config.h` and uncomment
   the line that matches your hardware unit.
3. **Ctrl+Shift+P → Particle: Configure for Device** → choose OS version and
   device name.
4. **Ctrl+Shift+P → Particle: Flash Application and Device OS (local)** for
   the very first flash. Answer **Y** to the USB-driver prompt; ignore any
   "missing permissions" messages.
5. After the first flash, subsequent flashes only need
   **Particle: Flash Application (local)**.

You are done. Skip to [After Setup — Sanity Check](#after-setup--sanity-check).

---

## Windows 11

Tested on the MINISFORUM Venus and HP Omen.

### 1. Git and GitHub Desktop

- Sign in to <https://github.com> in your browser first.
- Download GitHub Desktop from <https://desktop.github.com/> and install with
  defaults.
- Install Git for Windows from <https://git-scm.com/download/win> with default
  options.
- In GitHub Desktop: **File → Clone repository → URL**:

  ```
  https://github.com/davegutz/mySolarStateOfCharge
  ```

  Clone into `Documents\GitHub\`.

### 2. Python 3.12

- Download Python 3.12 from <https://www.python.org/>.
- Run the installer as a **non-admin install**.
- **Check** "Add Python to PATH".
- **Do not** tick "Disable path length limit".
- Open a new PowerShell after installation:

  ```powershell
  python --version
  python -m pip --version
  ```

### 3. VS Code and Particle Workbench (Windows)

- Download and run the VS Code installer from
  <https://code.visualstudio.com/>.
- Launch VS Code → Extensions panel → install **Particle Workbench**.
- Sign in to Particle (Particle icon in the activity bar → Log In).

Recommended (optional) extensions: **Python** (Microsoft), **Codeium AI**
(sign in via Google).

### 4. PyCharm Community

- Download from <https://www.jetbrains.com/pycharm/download/> and run the
  installer with defaults.
- After install, quit **JetBrains Toolbox** from the system tray and remove
  it from Startup (it is not needed).
- Open PyCharm → Open → browse to
  `Documents\GitHub\mySolarStateOfCharge\SOC_Particle\pyStateOfCharge\`.
- Settings → Project → Python Interpreter → Add → New Virtualenv. Base
  interpreter: the Python 3.12 you just installed.
- Let PyCharm index.
- Right-click `install.py` → Run.

### 5. puTTY

- Download the **64-bit MSI installer** from <https://www.putty.org/>.
- Run it with default settings.
- Add the puTTY install folder (typically `C:\Program Files\PuTTY\`) to the
  user `PATH`:
  - Start → "Edit the system environment variables" → Environment Variables →
    User variables → `Path` → Edit → New → paste the puTTY folder → OK.
- Restart any open shells / VS Code so the new `PATH` takes effect.

Find the COM port assigned to the Photon 2:

- **Device Manager → Ports (COM & LPT) → USB Serial Device (COM#)**

Create two saved puTTY sessions named `def` and `test` per the screenshots in
[dataReduction/putty/](dataReduction/putty/). Key settings:

- Connection type: **Serial**
- Serial line: `COM3` (or whatever Device Manager shows)
- Speed: **230400**
- Font: Courier New 10
- Logging: a local folder under `Documents\` (not OneDrive or Google Drive)

### 6. First flash (Windows)

1. Plug the Photon 2 into USB.
2. Open the project folder `SOC_Particle\` in VS Code.
3. Open `src\local_config.h` and uncomment your unit's include.
4. **Ctrl+Shift+P → Particle: Configure for Device** → pick OS and device.
5. **Ctrl+Shift+P → Particle: Flash Application and Device OS (local)** for
   the first flash. Confirm any USB-driver install prompt.

If you see the red "Unable to connect to the device" message, the device name
in `.vscode/settings.json` does not match what is on USB. Fix it; the flash
often succeeds anyway.

Done. Skip to [After Setup — Sanity Check](#after-setup--sanity-check).

---

## macOS

Tested on Intel and Apple Silicon Macs running recent macOS releases.

### 1. Xcode command-line tools and Homebrew

```bash
xcode-select --install
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2. Git and GitHub Desktop

- Sign in to <https://github.com> in your browser first.
- Download GitHub Desktop from <https://desktop.github.com/>.
- In GitHub Desktop: **File → Clone repository → URL**:

  ```
  https://github.com/davegutz/mySolarStateOfCharge
  ```

  Clone into `~/Documents/GitHub/`.

### 3. Python 3.12

```bash
brew install python@3.12
python3.12 --version
```

### 4. VS Code and Particle Workbench (macOS)

- Download VS Code for macOS from <https://code.visualstudio.com/>.
  Pick the build that matches your CPU (Apple Silicon or Intel).
- Drag VS Code to `/Applications/`.
- Launch → Extensions → install **Particle Workbench**.
- Sign in to Particle (Particle icon → Log In).

Alternative (per the original macOS notes): install Particle Workbench
directly from <https://docs.particle.io/quickstart/workbench/> and skip the
Azure IoT toolkit when offered.

### 5. PyCharm Community

- Download from <https://www.jetbrains.com/pycharm/download/> (Apple Silicon
  or Intel build).
- Drag PyCharm CE to `/Applications/`.
- Open PyCharm → Open → browse to
  `~/Documents/GitHub/mySolarStateOfCharge/SOC_Particle/pyStateOfCharge/`.
- Settings → Project → Python Interpreter → Add → New Virtualenv. Base
  interpreter: the Homebrew Python 3.12.
- Right-click `install.py` → Run.

For a packaged macOS `.app` build of the GUI, see
[doc/InstallationMacOS.md §Installing GUI](doc/InstallationMacOS.md#installing-gui)
— `install.py` produces an `.app` under `pyStateOfCharge/dist/` that can be
launched from Finder.

### 6. puTTY (via MacPorts + XQuartz)

puTTY on macOS needs both MacPorts and XQuartz:

1. Install **MacPorts** from <https://guide.macports.org/chunked/installing.macports.html>.
2. Install **XQuartz** from <https://www.xquartz.org/>.
3. Install puTTY:

   ```bash
   sudo port install putty
   ```

4. Launch puTTY (it will start XQuartz). In **XQuartz → Preferences → Input**
   enable:
   - Emulate 3-button mouse
   - Emulate syncing
   - Update Primary
5. To paste into the puTTY xterm window, use **Option + click**.

Find the serial device after plugging in the Photon 2:

```bash
ls -lha /dev/tty* > plugged.txt
# unplug
ls -lha /dev/tty* > np.txt
diff np.txt plugged.txt
```

The device usually appears as `/dev/tty.usbmodem*`.

Configure `def` and `test` puTTY sessions per the screenshots in
[dataReduction/putty/](dataReduction/putty/). Speed **230400**.

### 7. First flash (macOS)

Same as the other platforms — open `SOC_Particle/` in VS Code, set
`src/local_config.h`, **Particle: Configure for Device**, then
**Particle: Flash Application and Device OS (local)**.

---

## After Setup — Sanity Check

Once installation is complete on any platform:

1. With the Photon 2 plugged in, open puTTY and load the `def` session.
   Connect.
2. Type `vv1;` and press Enter. A stream of telemetry lines should appear.
3. Type `vv0;` to stop the stream. Type `h;` for the help menu.
4. In PyCharm, run `pyStateOfCharge/GUI_TestSOC.py`. The GUI opens.
5. Set the **Data Reduction Folder** to a local path.
6. Click **Init**, paste the clipboard contents into puTTY when prompted, and
   continue through **Start → Reset → Done → Compare**. See
   [User Interface](README.md#user-interface) and
   [doc/TestSOC.md](doc/TestSOC.md) for the full workflow.

If all of that works, the development environment is complete.

---

## Troubleshooting Quick Reference

| Symptom | Likely cause / fix |
| --- | --- |
| `STM32_Pin_Info does not name a type` build error | Wrong `#include` in `src/local_config.h` — pick the correct unit. |
| SOS 4 flashes (bus fault) after flash | SRAM exhausted — reduce `NSUM` in `constants.h`. |
| `FRAG` printed on serial | Heap fragmentation — reduce `NSUM` by ~8 below the value that just compiles. |
| Particle Workbench complains about `crc32` (Linux) | `sudo apt install libarchive-zip-perl` |
| VS Code Particle build silently fails (Linux) | You installed VS Code via `snap` or flatpak — uninstall it and use the `.deb` from code.visualstudio.com. |
| PyCharm can't find Python (Pop!_OS) | Flatpak sandbox — install PyCharm via `snap` instead. |
| "Unable to connect to the device" on flash (Windows) | `particle.targetDevice` in `.vscode/settings.json` mismatches USB device — fix the name; flash often succeeds anyway. |
| Converted current wanders ~10 min into a session | Laptop on AC adapter is biasing the ADC. **Do not** connect a laptop on AC power while monitoring; use battery only. |
| puTTY paste produces `. ? h` | In puTTY → Terminal → Keyboard, set **Enter Key Emulation = CR**. |
| Device shows wrong year (1999) | Photon's VBAT battery dead or device never synchronized — connect to Wi-Fi once via the Particle app, then use the GUI's `UT` button. |

For the exhaustive list see the [FAQ](README.md#faq) in the main README.
