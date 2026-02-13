import os
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


# Unix-like cat function
# e.g. > cat('out', ['in0', 'in1'], path_to_in='./')
def cat(out_file_name, in_file_names, in_path='./', out_path='./'):
    with open(os.path.join(out_path, out_file_name), 'w') as out_file:
        for in_file_name in in_file_names:
            with open(os.path.join(in_path, in_file_name)) as in_file:
                for line in in_file:
                    if line.strip():
                        out_file.write(line)
