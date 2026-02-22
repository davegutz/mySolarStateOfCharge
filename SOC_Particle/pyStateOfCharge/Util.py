import os
from pathlib import Path, PurePosixPath
import numpy.lib.recfunctions as rfn


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
            print(f"rename: {repl} already translated")
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


# Unix-like cat function
# e.g. > cat('out', ['in0', 'in1'], path_to_in='./')
def cat(out_file_name, in_file_names, in_path='./', out_path='./'):
    with open(str(PurePosixPath(out_path) / out_file_name), 'w') as out_file:
        for in_file_name in in_file_names:
            with open(str(PurePosixPath(in_path) / in_file_name)) as in_file:
                for line in in_file:
                    if line.strip():
                        out_file.write(line)
