# Copilot Instructions for mySolarStateOfCharge

## Project Overview

**mySolarStateOfCharge** is a battery State of Charge (SoC) monitor for LiFePO4 battery banks, built around the Particle Photon2 microcontroller. It combines Coulomb Counting with an Extended Kalman Filter (EKF) and dual current sensing to deliver accurate, self-calibrating energy monitoring with fault tolerance.

The repository contains:
- **SOC_Particle/** — Main firmware application (Photon2) + Python companion tools
- **SOC_Sense/** — Legacy sensor experiments
- **ble-uart-peripheral/** — BLE reference implementation
- **eeprom_test/** — EERAM test for older Argon boards

## Architecture

### High-Level Design

The system uses **three complementary estimation methods**:

1. **Coulomb Counting** — Integrates bipolar shunt current at 10 Hz to track charge directly
2. **Extended Kalman Filter (EKF)** — Estimates SoC from voltage + temperature; detects sensor faults
3. **Hysteresis Model** — Tracks VOC hysteresis of LiFePO4 to improve EKF accuracy and detect current sensor faults

**Sensor Redundancy:**
- Dual OPA333 op-amp circuits (HI-gain and LO-gain) convert ±0.075V shunt signal into 0-3.3V ADC range
- Combined with EKF, creates virtual triplex redundancy for current measurement
- Missing sensor or out-of-range signal detection + graceful degradation

**Hardware Interface (Particle Photon2):**
| Signal | Pin | Purpose |
|--------|-----|---------|
| Ib amp (primary) | A0 | OPA333 high-gain current, 1 Hz LPF |
| Vb battery voltage | A1 | 20k/4k7 divider + 1 Hz LPF |
| Ib noa (backup) | A2 | OPA333 low-gain current, 1 Hz LPF |
| Vc reference | A5 | Op-amp common voltage |
| Temperature | D3 | DS18B20 one-wire |
| Status LED | D7 | Heartbeat |
| USB / BLE | — | Serial + BLE notify |

**State Retention:**
- Fault snapshots + 30-minute history retained in SRAM
- Survives hard/soft resets via battery-backed VBAT pin
- Critical parameters (SoC, Coulomb count, hysteresis state) persist across power loss

### Firmware Architecture (SOC_Particle/src/)

**Core Classes:**
- `Battery` — Main SoC estimation; manages Coulomb Counter, EKF, and hysteresis
- `Chemistry_BMS` — Battery model parameters (chemistry, capacity, voltage curves)
- `Coulombs` — Coulomb Counter with calibration
- `Fault` — Fault detection state machine; signal selection logic
- `Sensors` — ADC sampling, filtering, and sensor diagnostics
- `Hysteresis` — VOC hysteresis lag model

**Key Control Flow:**
1. `setup()` initializes hardware, retained state, and models
2. `loop()` runs at fast cadence (10 Hz sensor read, 1 Hz filter update)
3. `Talk` interface processes serial commands for tweaking and testing
4. `SUMMARY_DELAY` (30 min) triggers state snapshot to retained SRAM

**Hardware Configuration:**
- Multiple device configs: `soc2p2_hi_lo.h`, `soc3p2_hi_lo.h`, `soc4p2_hi_lo.h`
- Selected in `local_config.h` via `#include`
- Each config specifies: hardware unit ID, sensor calibration, battery chemistry, fault thresholds
- Example: `#include "soc3p2_hi_lo.h"` (guest room 12V system)

### Python Companion Tools (SOC_Particle/pyStateOfCharge/)

**Primary GUI:** `GUI_TestSOC.py`
- Automates puTTY serial sessions and data capture
- Runs model simulation and overlots against hardware runs
- Buttons: Init → Start → Reset (timer) → Done (save) → Compare

**Key Utilities:**
- `Battery.py` — Python port of firmware battery model
- `CompareFault.py` — Overlay fault data and compare SoC estimates
- `CompareHistSim.py` — Compare captured history against simulation
- `CompareRunSim.py`, `CompareRunRun.py` — Run-vs-run and run-vs-sim plotting

**Data Flow:**
- puTTY captures serial stream to `.txt` files in `dataReduction/`
- GUI parses `.txt`, initializes model state, runs simulation
- Matplotlib overlots both for regression analysis

## Build, Test, and Lint

### Firmware (Particle Photon2)

**Build & Flash:**
```bash
# VS Code: Ctrl+Shift+P → Particle: Flash Application and Device OS (local)
# Command line (if particle-cli installed):
particle compile photon2 src/
particle flash <DEVICE_ID> target/photon2_platform_15.bin
```

**Configuration:**
- Select device in `src/local_config.h` (uncomment the correct `#include`)
- Run `Ctrl+Shift+P → Particle: Configure for Device` to set OS version + device name

**First Flash:**
- Requires USB drivers and Device OS flash (follow INSTALL.md prompts)
- Subsequent flashes use "Flash Application (local)" only

**Troubleshooting:**
- `STM32_Pin_Info does not name a type` → Wrong config in `local_config.h`
- `SOS 4 (bus fault)` → Reduce `NSUM` in `constants.h` (SRAM exceeded)
- Device not found → Try different USB port/cable, or put device in DFU mode (hold MODE + RESET until blinking yellow)

### Python

**Environment Setup:**
```bash
# PyCharm: Create venv using local Python interpreter
# Then run install.py from PyCharm to install packages
cd SOC_Particle/pyStateOfCharge/
python3 install.py
```

**Running GUI:**
```bash
# From PyCharm: open GUI_TestSOC.py and run
# Or: python3 GUI_TestSOC.py
```

**Data Analysis Scripts:**
- Individual comparison scripts can be run from PyCharm or command line
- All scripts expect `.txt` data files in `dataReduction/`

### Serial Interface Testing

**puTTY Sessions:**
Two saved sessions required:
1. **`def`** (default/idle) — For listening to heartbeat
2. **`test`** (active capture) — For data collection with logging

**Configuration (both sessions):**
- Connection type: Serial
- Speed: 230400 baud
- Logging path: local `dataReduction/` folder (not cloud storage)
- Font: Free Mono 10 (Linux) or Courier New 10 (Windows)

**Quick Test:**
```
vv1;     # Start data stream
vv0;     # Stop stream
h;       # Help/command list
```

## Key Conventions

### Naming and Configuration

**Device Configs:**
- Format: `{systemName}{photonVersion}{sensorType}.h`
- Example: `soc3p2_hi_lo.h` = SOC system, Photon2, HI_LO current sensors
- Always update corresponding entry in `local_config.h`

**Hardware Unit Identifiers:**
- Set via `#define HDWE_UNIT "soc3p2_hi_lo"` in device config
- Used in version string and Particle Cloud monitoring
- Must match device name in Particle console

**Calibration Parameters** (all in device config):
- `CURR_BIAS_AMP` / `CURR_BIAS_NOA` — Current sensor offsets (set via Talk: `DA`, `DB`)
- `VOLT_BIAS` — Voltage sensor bias (Talk: `Dc`)
- `TEMP_BIAS` — Temperature sensor bias (Talk: `Dt`)
- Marked with `*` in comments to indicate adjustable via Talk interface

### Fault Handling

**Fault Detection Strategy:**
- Each signal (Ib_amp, Ib_noa, Vb, Tb) continuously validated
- Out-of-range detection + signal disagreement tests
- Fault state → LED flash pattern (every 4th update minor, every 2nd update major)

**Signal Selection Logic:**
- Coulomb Counter + EKF can be fed from different signal subsets
- `CC_DIFF_SOC_DIS_THRESH` = threshold for EKF-vs-CC disagreement test
- FAKE_FAULTS mode: detect but don't act (for testing with noisy prototypes)

**Retained Parameters:**
- Fault snapshots and 30-minute history stored in SRAM
- Persist across resets; printed on boot via `Talk('Q')`
- `DISAB_VB_FA_LT`, `DISAB_TB_FA` flags disable certain fault latches (e.g., temperature sensor was noisy)

### Talk Interface

Serial command protocol for device interaction:

**Format:** `command parameters;` (terminator: `;`)

**Common Commands:**
- `h;` — Help
- `vv1;` / `vv0;` — Start/stop verbose output stream
- `soc N;` — Set SoC state to N%
- `Q;` — Query and print all retained state (used for regression analysis)
- `Rs;` — Attempt software filter reset
- `RR;` — Reset to installed configuration
- `DA X;`, `DB X;`, `Dc X;`, `Dt X;` — Set current amp bias, current noa bias, voltage bias, temp bias
- `Xm N;` — Set modeling mode (bitmap: 0=all hardware, 7=all modeled)
- `BS N;`, `BP N;` — Scale battery capacity in series/parallel

**Tweaking Philosophy:**
- All adjustments preserve `delta_q` (charge delta) between estimates
- System resets to saturation (0% charge at max voltage) whenever fully charged
- This is the "ground truth" for calibration

### Model and Physics

**Battery Model Components:**
1. **Voltage Lookup:** `voc(soc, temp)` — Industry standard LiFePO4 OCV curve (tables)
2. **Series Resistance:** `Rss(soc, temp)` — Temperature-dependent ohmic resistance
3. **Randles RC:** Two-stage RC network for transient response
4. **Hysteresis:** VOC lag from diffusion + charge redistribution

**Temperature Compensation:**
- Capacity scales ±0.01 fractional per degree C from nominal 25°C
- Both Coulomb Counter and EKF track temperature-dependent capacity

**Assumptions (from requirements):**
- Perfect sensor calibration (errors tracked as accuracy statement)
- Randles 2-state RC + hysteresis RC model
- All current entering model ends up as charge (transient energy stored but accounted for)
- Battery has Coulombic charging efficiency (usually >98% for LiFePO4)

### File Organization

**`src/` Structure:**
- `.ino` file: Main sketch (setup, loop)
- `*.h` / `*.cpp` pairs: Class implementations (Battery, Fault, Sensors, etc.)
- `Adafruit/` → Third-party OLED display + I2C drivers (external)
- `myLibrary/` → Custom EKF, state-space, iteration solvers
- Hardware-specific configs: `local_config.h`, `soc*p*.h`

**`pyStateOfCharge/` Structure:**
- Root: Main GUI + install script
- `filter/` → Python EKF/Kalman implementations
- `figures/` → Generated comparison plots (git-ignored)
- `g{YYYYMMDD}{revision}/` → Dated model/test data snapshots

**Data Reduction:**
- `dataReduction/` → puTTY captured `.txt` files + puTTY session configs
- Expected format: timestamp-prefixed lines from Talk stream
- Old data backed up into dated subdirectories

### Constants and Thresholds

**Timing (all in ms):**
- `TALK_DELAY` = 313 ms (serial input polling)
- `READ_DELAY` = 100 ms (sensor ADC sampling)
- `PUBLISH_SERIAL_DELAY` = 400 ms (serial output cadence)
- `SUMMARY_DELAY` = 1800000 ms (30 min state snapshot)
- `SNAP_WAIT` = 10000 ms (fault snapshot interval)

**Sensor Ranges:**
- `IB_ABS_MAX_AMP` = ~12 A (HI-gain amplifier max)
- `IB_ABS_MAX_NOA` = ~78.5 A (LO-gain amplifier max)
- `TEMP_RANGE_CHECK` = -5°C (minimum valid), `TEMP_RANGE_CHECK_MAX` = 70°C

**Fault Thresholds:**
- `CC_DIFF_SOC_DIS_THRESH` = 0.5% (Coulomb Counter vs. EKF disagreement)
- `FI_NOM` / `FO_NOM` = 200% (wrap detection scalar for signal divergence)

## Installation and Setup

See **[SOC_Particle/INSTALL.md](../SOC_Particle/INSTALL.md)** for complete setup instructions covering:
- Git and GitHub Desktop
- VS Code + Particle Workbench extension (with crc32 tool on Linux)
- PyCharm + Python venv setup
- puTTY + serial permissions + session configuration
- First flash to Photon2
- Running GUI_TestSOC.py

Platform-specific guides:
- Windows: [doc/InstallationWindows.md](../SOC_Particle/doc/InstallationWindows.md)
- Linux: [doc/InstallationLinux.md](../SOC_Particle/doc/InstallationLinux.md)
- macOS: [doc/InstallationMacOS.md](../SOC_Particle/doc/InstallationMacOS.md)

## Common Workflows

### Add Support for New Battery Chemistry

1. Create new chemistry file: `src/chemistry_newbrand.h` (copy from existing, adjust curves)
2. Add entry to `Chemistry_BMS::Chemistry_BMS()` constructor
3. Update `#define CHEM` in device config
4. Recompile and test with Talk interface: `Q;` to verify parameters

### Run Regression Test

1. Flash device with target firmware
2. Open puTTY `test` session and clear log file
3. Run device through known cycle (e.g., controlled discharge)
4. Close puTTY and save `.txt` file
5. Open GUI_TestSOC.py → set data folder → Compare button
6. Overlay against previous reference run or simulation

### Calibrate Current Sensor

1. Device must be at rest (zero current)
2. Run `DA X;` (HI-gain) and `DB X;` (LO-gain) to remove offsets
3. Record values in device config for future compiles
4. Optionally: Run Talk `Q;` and inspect retained state for accuracy

### Debug Fault Detection

1. Set `FAKE_FAULTS true` in device config to detect without disabling signals
2. Inject bad signal via Talk (e.g., `vv1;` to stream and inspect values)
3. Monitor LED flash pattern (period indicates severity)
4. Print retained state with `Q;` to see fault log and signal selection choices
5. Adjust thresholds (FI_NOM, FO_NOM, CC_DIFF_SOC_DIS_THRESH) if too sensitive

## References

- **[README.md](../README.md)** — Project overview, hardware table, repository structure
- **[SOC_Particle/README.md](../SOC_Particle/README.md)** — Firmware architecture, requirements, assumptions
- **[SOC_Particle/INSTALL.md](../SOC_Particle/INSTALL.md)** — Development environment setup
- **[SOC_Particle/DecisionTables.md](../SOC_Particle/DecisionTables.md)** — Fault state machine logic
- Datasheets: `datasheets/` folder (OPA333, Photon2, DS18B20, ASD1013 shunt)
- Theory: `Battery State/` folder (EKF derivations, sandbox models)
