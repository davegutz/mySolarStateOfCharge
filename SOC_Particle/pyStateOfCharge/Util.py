from pathlib import PurePosixPath
import numpy as np
import numpy.lib.recfunctions as rfn
import csv
import os


# rename array elements, either class or recarray
def rename(ra, targ, repl):
    try:
        ra = rfn.rename_fields(ra, {targ: repl})
    except ValueError:
        # Rename
        if ra.dtype.names.__contains__(targ):
            names = list(ra.dtype.names)
            names[names.index(targ)] = repl
            ra.dtype.names = tuple(names)
    except AttributeError:
        if hasattr(ra, targ):
            val = getattr(ra, targ)
            setattr(ra, repl, val)
            delattr(ra, targ)
        elif hasattr(ra, repl):
            # print(f"rename: {repl} already translated")
            pass
        else:
            print(f"rename:  neither {targ} nor {repl} found")
    return ra

# One place to rename them all
def rename_all(ra_or_cl):
    ra_or_cl = rename(ra_or_cl, 'Fx_', 'Fx')
    ra_or_cl = rename(ra_or_cl, 'Bu_', 'Bu')
    ra_or_cl = rename(ra_or_cl, 'Q_', 'Q')
    ra_or_cl = rename(ra_or_cl, 'R_', 'R')
    ra_or_cl = rename(ra_or_cl, 'P_', 'P')
    ra_or_cl = rename(ra_or_cl, 'S_', 'S')
    ra_or_cl = rename(ra_or_cl, 'K_', 'K')
    ra_or_cl = rename(ra_or_cl, 'u_', 'u')
    ra_or_cl = rename(ra_or_cl, 'x_', 'x')
    ra_or_cl = rename(ra_or_cl, 'y_', 'y')
    ra_or_cl = rename(ra_or_cl, 'z_', 'z')
    ra_or_cl = rename(ra_or_cl, 'x_prior_', 'x_prior')
    ra_or_cl = rename(ra_or_cl, 'frz_', 'frz')
    ra_or_cl = rename(ra_or_cl, 'P_prior_', 'P_prior')
    ra_or_cl = rename(ra_or_cl, 'x_post_', 'x_post')
    ra_or_cl = rename(ra_or_cl, 'P_post_', 'P_post')
    ra_or_cl = rename(ra_or_cl, 'hx_', 'hx')
    ra_or_cl = rename(ra_or_cl, 'H_', 'H')
    ra_or_cl = rename(ra_or_cl, 'tb_f_hx_', 'tb_f_for_hx')
    ra_or_cl = rename(ra_or_cl, 'x_for_hx_', 'x_for_hx')
    ra_or_cl = rename(ra_or_cl, 'skip', 'skip_temp')

    return ra_or_cl


def save_struct_to_csv(obj, file_path):
    """Write every time-dependent variable in obj to a CSV file.

    Discovers the time axis by looking for a 1-D numpy array named 'time',
    then 'cTime'.  Every other 1-D numpy array whose length matches the time
    axis is written as a column.  The time column is always first; the rest
    are alphabetically sorted.  Scalars, multi-dimensional arrays, and arrays
    of a different length are silently skipped.

    Works with Python class instances (uses __dict__) and numpy recarrays
    (uses dtype.names).  obj=None is silently skipped.
    """
    if obj is None:
        print(f"save_struct_to_csv: obj is None, skipping {file_path}")
        return

    # --- collect raw attribute dict ------------------------------------------
    if hasattr(obj, 'dtype') and getattr(obj.dtype, 'names', None):
        # numpy recarray
        attrs = {name: np.atleast_1d(obj[name]) for name in obj.dtype.names}
    elif hasattr(obj, '__dict__'):
        attrs = vars(obj)
    else:
        attrs = {a: getattr(obj, a) for a in dir(obj) if not a.startswith('_')}

    # --- find time axis -------------------------------------------------------
    time = None
    t_col = None
    for candidate in ('time', 'cTime'):
        raw = attrs.get(candidate)
        if raw is None and hasattr(obj, candidate):
            raw = getattr(obj, candidate)
        if raw is not None:
            arr = np.asarray(raw)
            if arr.ndim == 1 and len(arr) > 1:
                time = arr
                t_col = candidate
                break

    if time is None:
        print(f"save_struct_to_csv: no time array found in {type(obj).__name__}, skipping {file_path}")
        return

    n = len(time)

    # --- collect all matching 1-D arrays -------------------------------------
    cols = {}
    for name, val in attrs.items():
        if name.startswith('_') or callable(val):
            continue
        try:
            arr = np.asarray(val)
            if arr.ndim == 1 and len(arr) == n and arr.dtype.kind in 'fiubc':
                cols[name] = arr
        except Exception:
            pass

    if t_col not in cols:
        cols[t_col] = time

    headers = [t_col] + sorted((k for k in cols if k != t_col), key=str.lower)

    # --- write CSV -----------------------------------------------------------
    out_dir = os.path.dirname(file_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for i in range(n):
            writer.writerow([cols[h][i] for h in headers])

    print(f"save_struct_to_csv: {len(headers)} cols x {n} rows -> {file_path}")


# Unix-like cat function
# e.g. > cat('out', ['in0', 'in1'], path_to_in='./')
def cat(out_file_name, in_file_names, in_path='./', out_path='./'):
    with open(str(PurePosixPath(out_path) / out_file_name), 'w') as out_file:
        for in_file_name in in_file_names:
            with open(str(PurePosixPath(in_path) / in_file_name)) as in_file:
                for line in in_file:
                    if line.strip():
                        out_file.write(line)
