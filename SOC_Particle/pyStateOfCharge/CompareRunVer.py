"""Compare all run/ver CSV pairs in the temp folder for the version pointed to by the Plink .ini file.

For each *_run.csv / *_ver.csv pair, loads both into pandas, filters time >= 0, and reports
numeric columns where |run - ver| > tol + rtol*peak, with `peak` = column-wise max(|run|, |ver|).
Combined absolute/relative criterion scaled to a signal's full-range magnitude — large-swing
signals (e.g. q_capacity ~3.5e5) aren't tripped by tiny absolute differences, and quiet
near-zero transients on a big-swing signal aren't flagged either.  Produces a pyplot figure
for each pair, with up to 9 subplots per figure.

Usage:
    python CompareRunVer.py [--ini PATH] [--tol FLOAT] [--rtol FLOAT] [--version VERSION]
"""

import argparse
import math
import os
import platform
import re
import sys
from configparser import ConfigParser
from pathlib import Path, PurePosixPath
import matplotlib.pyplot as plt
import pandas as pd
from PlotKiller import show_killer
from local_paths import version_from_data_file, local_paths

# Default EKF frame-rate multiplier — must match firmware EKF_EFRAME_MULT in src/Battery.h
EKF_DEFAULT_FRAME_MULT = 20

# Column names suggesting EKF outputs/state (soc_ekf, voc_ekf, y_ekf, kfres, kf_v_m, *_kf, ekf_reset, flt_ekf, ...)
_EKF_COL_RE = re.compile(r'(?:^|_)(?:ekf|kf|kfres|kfres_1)(?:_|$)', re.IGNORECASE)

# Column names containing "Tb" (temperature) — skipped whenever reset_temp is True
_TB_COL_RE = re.compile(r'Tb', re.IGNORECASE)


def _extract_ed(macro_str):
    """Return int ED from a macro fragment like 'Dr400;ED1;DP1;', or None if absent."""
    if not macro_str:
        return None
    m = re.search(r'ED(\d+)', macro_str)
    return int(m.group(1)) if m else None


def _test_name_from_run_filename(run_name):
    """'rapidTweakRegression_pro1a_bb_..._run.csv' -> 'rapidTweakRegression'."""
    stem = run_name[:-len('_run.csv')] if run_name.endswith('_run.csv') else run_name
    return stem.split('_', 1)[0]


def _get_ed_for_case(test_name):
    """Look up test_name in GUI_common.lookup, parse ED from its macro, default to EKF_DEFAULT_FRAME_MULT."""
    if not test_name:
        return EKF_DEFAULT_FRAME_MULT
    try:
        from GUI_common import lookup
    except Exception:
        return EKF_DEFAULT_FRAME_MULT
    entry = lookup.get(test_name)
    if not entry or len(entry) < 2:
        return EKF_DEFAULT_FRAME_MULT
    ed = _extract_ed(entry[1])
    return ed if ed is not None else EKF_DEFAULT_FRAME_MULT


def _is_ekf_column(name):
    return bool(_EKF_COL_RE.search(name))


def _is_tb_column(name):
    return bool(_TB_COL_RE.search(name))

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

def compare_pair(run_path, ver_path, tol, rtol=1e-3, ed=None):
    """Return a summary dict for one run/ver pair.

    A sample is flagged when |run - ver| > tol + rtol * peak, where `peak` is the
    column-wise max(|run|, |ver|) across all rows.  Tolerance scales with the
    signal's full-range magnitude — large-swing signals (e.g. q_capacity ~3.5e5)
    aren't tripped by tiny absolute differences, and transients near zero on a
    big-swing signal aren't flagged just because the per-row reference collapses.

    `ed` is the EKF frame-rate multiplier (talk param "ED").  When ed > 1, the EKF
    needs ~ed*dt to initialize, so EKF-related columns are not flagged before that.
    If None, ed is inferred from the run filename via GUI_common.lookup.
    """
    try:
        df_run = pd.read_csv(run_path)
        df_ver = pd.read_csv(ver_path)
    except Exception as e:
        return {'file': run_path.name, 'ver_file': ver_path.name, 'error': str(e), 'diffs': []}

    # filter time >= 0
    if 'time' not in df_run.columns or 'time' not in df_ver.columns:
        return {'file': run_path.name, 'ver_file': ver_path.name, 'error': 'no "time" column', 'diffs': []}

    df_run = df_run[pd.to_numeric(df_run['time'], errors='coerce') >= 0].copy()
    df_ver = df_ver[pd.to_numeric(df_ver['time'], errors='coerce') >= 0].copy()

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
        return {'file': run_path.name, 'ver_file': ver_path.name,
                'error': 'no rows with time >= 0 before reset', 'diffs': []}
    df_run = df_run.iloc[:n].reset_index(drop=True)
    df_ver = df_ver.iloc[:n].reset_index(drop=True)

    # numeric columns present in both
    shared_cols = [c for c in df_run.columns if c in df_ver.columns]
    numeric_cols = [c for c in shared_cols
                    if pd.api.types.is_numeric_dtype(df_run[c]) and pd.api.types.is_numeric_dtype(df_ver[c])
                    and not pd.api.types.is_bool_dtype(df_run[c]) and not pd.api.types.is_bool_dtype(df_ver[c])]

    if ed is None:
        ed = _get_ed_for_case(_test_name_from_run_filename(run_path.name))
    dt_med = float(df_run['time'].diff().median()) if len(df_run) > 1 else 0.0
    if not (dt_med > 0):
        dt_med = 0.0
    ekf_skip_until = (ed * dt_med) if (ed and ed > 1) else 0.0

    # Build a boolean mask for rows where reset_temp is active (True/1), used to suppress Tb checks.
    reset_temp_mask = None
    for _rt_col in ('reset_temp', 'rt'):
        if _rt_col in df_run.columns:
            reset_temp_mask = df_run[_rt_col].astype(float).astype(bool)
            break
        if _rt_col in df_ver.columns:
            reset_temp_mask = df_ver[_rt_col].astype(float).astype(bool)
            break

    diffs = []
    for col in numeric_cols:
        delta = (df_run[col] - df_ver[col]).abs()
        if ekf_skip_until > 0.0 and _is_ekf_column(col):
            delta = delta.where(df_run['time'] >= ekf_skip_until, 0.0)
        if reset_temp_mask is not None and _is_tb_column(col):
            delta = delta.where(~reset_temp_mask, 0.0)
        peak_run = df_run[col].abs().max()
        peak_ver = df_ver[col].abs().max()
        peak = max(peak_run if pd.notna(peak_run) else 0., peak_ver if pd.notna(peak_ver) else 0.)
        threshold = tol + rtol * peak
        bad = delta[delta > threshold]
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
        'ver_file': ver_path.name,
        'n_rows': n,
        'diffs': sorted(diffs, key=lambda d: d['max_diff'], reverse=True),
        'run_only': run_only_cols,
        'df_run': df_run,
        'df_ver': df_ver,
        'ed': ed,
        'ekf_skip_until': ekf_skip_until,
    }


# ── report ────────────────────────────────────────────────────────────────────

def report(results, tol, rtol=1e-3, option='', macro=''):
    any_diff = any(r.get('diffs') for r in results)
    print(f"\n{'='*72}")
    print(f"  CompareRunVer  |  tol={tol}  rtol={rtol}  |  option={option}  |  macro={macro}  |  {len(results)} pair(s)")
    print(f"{'='*72}\n")

    for r in results:
        run_file = r['file']
        ver_file = r.get('ver_file', run_file.replace('_run.csv', '_ver.csv'))
        pair_label = f"{run_file}  vs  {ver_file}"
        if 'error' in r:
            print(f"  {pair_label}")
            print(f"    ERROR: {r['error']}\n")
            continue
        run_only = r.get('run_only', [])
        ed_note = ''
        if r.get('ekf_skip_until', 0.0) > 0.0:
            ed_note = f"  [EKF init skip: t<{r['ekf_skip_until']:.3f}s, ED={r.get('ed')}]"
        if not r['diffs']:
            print(f"  {pair_label}  — no differences > tol={tol} + rtol={rtol}*peak  ({r['n_rows']} rows){ed_note}")
            if run_only:
                print(f"    run_only ({len(run_only)}): {', '.join(run_only)}")
            print()
            continue
        print(f"  {pair_label}  ({r['n_rows']} rows, {len(r['diffs'])} differing param(s)){ed_note}")
        if run_only:
            print(f"    Parameters in _run only ({len(run_only)}): {', '.join(run_only)}")
        print(f"    {'param':<30}  {'n_bad':>6}  {'max|d|':>12}  {'mean|d|':>12}  {'first_t':>10}")
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


def plot_diffs(results, data_file=None, save_plots=True, terse=False, hardcopy=True, show_killer_=True):
    """For each pair with differences, produce figure(s) with ≤9 run-vs-ver subplots.

    When show_killer_=False, returns (fig_list, fig_files, save_pdf_path) without blocking
    so the caller can accumulate figures across multiple cases and call show_batch_diffs once.
    """
    fig_list = []
    fig_files = []
    save_pdf_path = None
    for r in results:
        if 'error' in r or not r.get('diffs'):
            continue

        df_run = r['df_run']
        df_ver = r['df_ver']
        t_run = df_run['time'].values
        t_ver = df_ver['time'].values
        run_file = r['file']
        ver_file = r.get('ver_file', run_file.replace('_run.csv', '_ver.csv'))
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
            fig_label = f"  [{fig_idx + 1}/{n_figs}]" if n_figs > 1 else ""
            fig.suptitle(f"{run_file}  vs  {ver_file}{fig_label}", fontsize=9)

            for sub_idx, d in enumerate(batch):
                ax = axes[sub_idx // COLS][sub_idx % COLS]
                param = d['param']
                ax.plot(t_run, df_run[param].values, label='run', linewidth=1)
                ax.plot(t_ver, df_ver[param].values, label='ver', linewidth=1, linestyle='--')
                ax.set_title(f"{param}\nmax|d|={d['max_diff']:.4g}", fontsize=8)
                ax.set_xlabel('time (s)', fontsize=7)
                ax.tick_params(labelsize=7)
                ax.legend(fontsize=7, loc='best')
                ax.grid(True, linewidth=0.4)
            fig_file_name = os.path.join(save_pdf_path, 'CompareRunVer_' + str(len(fig_list)) + ".png")
            fig_files.append(fig_file_name)
            if save_plots and not terse:
                plt.savefig(fig_file_name, format="png")

            # hide unused axes in last row
            for empty_idx in range(n_sub, n_rows * COLS):
                axes[empty_idx // COLS][empty_idx % COLS].set_visible(False)

            fig.tight_layout(rect=(0, 0, 1, 0.95))

    if not fig_list:
        return ([], [], None) if not show_killer_ else None

    plt.show(block=False)

    if not show_killer_:
        return fig_list, fig_files, save_pdf_path

    string = 'plots ' + str(fig_list[0].number) + ' - ' + str(fig_list[-1].number)
    show_killer(string, 'CompareRunSim', fig_list=fig_list, fig_files=fig_files, pdf_path=save_pdf_path,
                pdf_base=os.path.join(save_pdf_path, 'CompareRunVer'), hardcopy=hardcopy)


def show_batch_diffs(all_fig_list, all_fig_files, save_pdf_path, hardcopy=True):
    """Open PlotKiller once with all figures accumulated across a batch of cases."""
    if not all_fig_list:
        return
    string = 'plots ' + str(all_fig_list[0].number) + ' - ' + str(all_fig_list[-1].number)
    show_killer(string, 'CompareRunSim', fig_list=all_fig_list, fig_files=all_fig_files,
                pdf_path=save_pdf_path,
                pdf_base=os.path.join(save_pdf_path, 'CompareRunVer'), hardcopy=hardcopy)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ini', default=None, help='path to GUI_PlinkSOC .ini file (auto-detected by default)')
    parser.add_argument('--tol', type=float, default=1e-3, help='absolute difference tolerance (default 1e-3)')
    parser.add_argument('--rtol', type=float, default=1e-3,
                        help='relative tolerance: flag if |run-ver| > tol + rtol*peak, peak=column-wise max(|run|,|ver|) '
                             '(default 1e-3, ~3 sig digits)')
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

    results = [compare_pair(run, ver, args.tol, args.rtol) for run, ver in pairs]
    report(results, args.tol, args.rtol, option=option, macro=macro)
    plot_diffs(results)


if __name__ == '__main__':
    main()
