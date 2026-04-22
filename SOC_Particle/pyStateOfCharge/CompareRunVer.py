"""Compare all run/ver CSV pairs in the temp folder for the version pointed to by the Plink .ini file.

For each *_run.csv / *_ver.csv pair, loads both into pandas, filters time >= 0, and reports
numeric columns where |run - ver| > 1e-3 (disagreements visible at 3 decimal places).
Produces a pyplot figure for each pair, with up to 9 subplots per figure.

Usage:
    python CompareRunVer.py [--ini PATH] [--tol FLOAT] [--version VERSION]
"""

import argparse
import math
import os
import platform
import sys
from configparser import ConfigParser
from pathlib import Path, PurePosixPath
import matplotlib.pyplot as plt
import pandas as pd
from PlotKiller import show_killer
from local_paths import version_from_data_file, local_paths

# ── locate the .ini file ──────────────────────────────────────────────────────

def ini_path():
    plat = platform.system()
    if plat == 'Linux':
        return '/home/daveg/.local/GUI_PlinkSOC_linux.ini'
    elif plat == 'Darwin':
        return '/Users/daveg/.local/GUI_PlinkSOC_macos.ini'
    else:
        local = os.getenv('LOCALAPPDATA') or str(Path.home() / 'AppData' / 'Local')
        return str(Path(local) / 'GUI_PlinkSOC.ini')


def read_ini(ini_file):
    """Return (version, option, macro) from the Plink .ini file."""
    cfg = ConfigParser()
    cfg.read(ini_file)
    version = cfg['test']['version']
    option = cfg['others'].get('option', '')
    macro = cfg['others'].get('macro', '')
    return version, option, macro


# ── locate the temp folder ────────────────────────────────────────────────────

def temp_folder(version):
    plat = platform.system()
    if plat == 'Linux':
        base = '/home/daveg/.local/SOC_Particle'
    elif plat == 'Darwin':
        base = '/Users/daveg/.local/SOC_Particle'
    else:
        base = str(Path(os.getenv('LOCALAPPDATA') or '.') / 'SOC_Particle')
    return str(PurePosixPath(base) / version / 'temp')


# ── find run/ver pairs ────────────────────────────────────────────────────────

def find_pairs(temp_dir, option=''):
    """Return (run_path, ver_path) tuples for every *_run.csv with a matching *_ver.csv.

    If option is non-empty, only include pairs whose filename starts with that option string.
    """
    pairs = []
    for p in sorted(Path(temp_dir).glob('*_run.csv')):
        if option and not p.name.startswith(option + '_'):
            continue
        ver = Path(str(p).replace('_run.csv', '_ver.csv'))
        if ver.is_file():
            pairs.append((p, ver))
    return pairs


# ── compare a single pair ─────────────────────────────────────────────────────

def compare_pair(run_path, ver_path, tol):
    """Return a summary dict for one run/ver pair."""
    try:
        df_run = pd.read_csv(run_path)
        df_ver = pd.read_csv(ver_path)
    except Exception as e:
        return {'file': run_path.name, 'error': str(e), 'diffs': []}

    # filter time >= 0
    if 'time' not in df_run.columns or 'time' not in df_ver.columns:
        return {'file': run_path.name, 'error': 'no "time" column', 'diffs': []}

    df_run = df_run[pd.to_numeric(df_run['time'], errors='coerce') > 0].copy()
    df_ver = df_ver[pd.to_numeric(df_ver['time'], errors='coerce') > 0].copy()

    # clip to before the first reset event in ver that happens later in the run (time > 1 s).
    # Skip the clip if it would leave fewer than 5 rows (degenerate / old-format files).
    if 'reset' in df_ver.columns:
        reset_mask = (df_ver['reset'].astype(str).str.lower().isin(['true', '1', '1.0'])
                      & (df_ver['time'] > 1.0))
        reset_rows = df_ver[reset_mask]
        if not reset_rows.empty:
            t_reset = float(reset_rows['time'].iloc[0])
            clipped_run = df_run[df_run['time'] < t_reset]
            clipped_ver = df_ver[df_ver['time'] < t_reset]
            if min(len(clipped_run), len(clipped_ver)) >= 5:
                df_run = clipped_run.copy()
                df_ver = clipped_ver.copy()

    # align on time via index reset (both should have identical row counts post-filter)
    n = min(len(df_run), len(df_ver))
    if n == 0:
        return {'file': run_path.name, 'error': 'no rows with time > 0 before reset', 'diffs': []}
    df_run = df_run.iloc[:n].reset_index(drop=True)
    df_ver = df_ver.iloc[:n].reset_index(drop=True)

    # numeric columns present in both
    shared_cols = [c for c in df_run.columns if c in df_ver.columns]
    numeric_cols = [c for c in shared_cols
                    if pd.api.types.is_numeric_dtype(df_run[c]) and pd.api.types.is_numeric_dtype(df_ver[c])
                    and not pd.api.types.is_bool_dtype(df_run[c]) and not pd.api.types.is_bool_dtype(df_ver[c])]

    diffs = []
    for col in numeric_cols:
        delta = (df_run[col] - df_ver[col]).abs()
        bad = delta[delta > tol]
        if bad.empty:
            continue
        diffs.append({
            'param': col,
            'n_bad': int(bad.count()),
            'max_diff': float(bad.max()),
            'mean_diff': float(bad.mean()),
            'first_time': float(df_run.loc[bad.index[0], 'time']),
        })

    run_only_cols = [c for c in df_run.columns if c not in df_ver.columns]

    return {
        'file': run_path.name,
        'n_rows': n,
        'diffs': sorted(diffs, key=lambda d: d['max_diff'], reverse=True),
        'run_only': run_only_cols,
        'df_run': df_run,
        'df_ver': df_ver,
    }


# ── report ────────────────────────────────────────────────────────────────────

def report(results, tol, option='', macro=''):
    any_diff = any(r.get('diffs') for r in results)
    print(f"\n{'='*72}")
    print(f"  CompareRunVer  |  tol={tol}  |  option={option}  |  macro={macro}  |  {len(results)} pair(s)")
    print(f"{'='*72}\n")

    for r in results:
        stem = r['file'].replace('_run.csv', '')
        if 'error' in r:
            print(f"  {stem}")
            print(f"    ERROR: {r['error']}\n")
            continue
        run_only = r.get('run_only', [])
        if not r['diffs']:
            print(f"  {stem}  — no differences > {tol}  ({r['n_rows']} rows)")
            if run_only:
                print(f"    run_only ({len(run_only)}): {', '.join(run_only)}")
            print()
            continue
        print(f"  {stem}  ({r['n_rows']} rows, {len(r['diffs'])} differing param(s))")
        if run_only:
            print(f"    Parameters in _run only ({len(run_only)}): {', '.join(run_only)}")
        print(f"    {'param':<30}  {'n_bad':>6}  {'max|Δ|':>12}  {'mean|Δ|':>12}  {'first_t':>10}")
        print(f"    {'-'*30}  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*10}")
        for d in r['diffs']:
            print(f"    {d['param']:<30}  {d['n_bad']:>6}  {d['max_diff']:>12.6f}  {d['mean_diff']:>12.6f}  {d['first_time']:>10.3f}")
        print()

    if not any_diff:
        print("  All pairs agree within tolerance.\n")


# ── plots ─────────────────────────────────────────────────────────────────────

COLS = 3
ROWS = 3
PER_FIG = COLS * ROWS


def plot_diffs(results, data_file=None,  save_plots=True, terse=False, hardcopy=True):
    """For each pair with differences, produce figure(s) with ≤9 run-vs-ver subplots."""
    fig_list = []
    fig_files = []
    for r in results:
        if 'error' in r or not r.get('diffs'):
            continue

        df_run = r['df_run']
        df_ver = r['df_ver']
        t_run = df_run['time'].values
        t_ver = df_ver['time'].values
        stem = r['file'].replace('_run.csv', '')
        diffs = r['diffs']
        n_figs = math.ceil(len(diffs) / PER_FIG)
        version = version_from_data_file(data_file)
        _, save_pdf_path, _ = local_paths(version)

        for fig_idx in range(n_figs):
            batch = diffs[fig_idx * PER_FIG:(fig_idx + 1) * PER_FIG]
            n_sub = len(batch)
            # keep grid rectangular: fill rows top-to-bottom
            n_rows = math.ceil(n_sub / COLS)
            fig, axes = plt.subplots(n_rows, COLS, figsize=(5 * COLS, 3 * n_rows), squeeze=False)
            fig_list.append(fig)
            fig_label = f"fig {fig_idx + 1}/{n_figs}" if n_figs > 1 else ""
            fig.suptitle(f"{stem}  {fig_label}", fontsize=9)

            for sub_idx, d in enumerate(batch):
                ax = axes[sub_idx // COLS][sub_idx % COLS]
                param = d['param']
                ax.plot(t_run, df_run[param].values, label='run', linewidth=1)
                ax.plot(t_ver, df_ver[param].values, label='ver', linewidth=1, linestyle='--')
                ax.set_title(f"{param}\nmax|Δ|={d['max_diff']:.4g}", fontsize=8)
                ax.set_xlabel('time (s)', fontsize=7)
                ax.tick_params(labelsize=7)
                ax.legend(fontsize=7, loc='best')
                ax.grid(True, linewidth=0.4)
            fig_file_name = 'CompareRunVer' + '_' + str(len(fig_list)) + ".png"
            fig_files.append(fig_file_name)
            if save_plots and not terse:
                plt.savefig(fig_file_name, format="png")

            # hide unused axes in last row
            for empty_idx in range(n_sub, n_rows * COLS):
                axes[empty_idx // COLS][empty_idx % COLS].set_visible(False)

            fig.tight_layout(rect=(0, 0, 1, 0.95))

    if any(r.get('diffs') for r in results):
        plt.show(block=False)

    string = 'plots ' + str(fig_list[0].number) + ' - ' + str(fig_list[-1].number)
    show_killer(string, 'CompareRunSim', fig_list=fig_list, fig_files=fig_files, pdf_path=save_pdf_path,
                pdf_base=save_pdf_path, hardcopy=hardcopy)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ini', default=None, help='path to GUI_PlinkSOC .ini file (auto-detected by default)')
    parser.add_argument('--tol', type=float, default=1e-3, help='difference tolerance (default 1e-3)')
    parser.add_argument('--version', default=None, help='override version string from .ini')
    args = parser.parse_args()

    ini_file = args.ini or ini_path()
    if not Path(ini_file).is_file():
        print(f"ERROR: .ini file not found: {ini_file}", file=sys.stderr)
        sys.exit(1)

    version, option, macro = read_ini(ini_file)
    if args.version:
        version = args.version
    print(f"ini:      {ini_file}")
    print(f"version:  {version}")
    print(f"option:   {option}")
    print(f"macro:    {macro}")

    temp_dir = temp_folder(version)
    print(f"temp:     {temp_dir}")

    if not Path(temp_dir).is_dir():
        print(f"ERROR: temp folder not found: {temp_dir}", file=sys.stderr)
        sys.exit(1)

    pairs = find_pairs(temp_dir, option=option)
    if not pairs:
        print(f"No run/ver pairs found in {temp_dir}")
        sys.exit(0)

    results = [compare_pair(run, ver, args.tol) for run, ver in pairs]
    report(results, args.tol, option=option, macro=macro)
    plot_diffs(results)


if __name__ == '__main__':
    main()
