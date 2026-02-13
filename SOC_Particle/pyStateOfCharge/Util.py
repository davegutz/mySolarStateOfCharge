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

# One place to rename them all
def rename_all(ra_or_cl):

    ra_or_cl = rename(ra_or_cl, 'dv_dm', 'dv_dyn_m')
    ra_or_cl = rename(ra_or_cl, 'dv_dn', 'dv_dyn_n')
    ra_or_cl = rename(ra_or_cl, 'e_w', 'e_wrap')
    ra_or_cl = rename(ra_or_cl, 'e_w_f', 'e_wrap_filt')
    ra_or_cl = rename(ra_or_cl, 'e_wm', 'e_wrap_m')
    ra_or_cl = rename(ra_or_cl, 'e_wm_f', 'e_wrap_m_filt')
    ra_or_cl = rename(ra_or_cl, 'e_wm_r', 'e_wrap_m_reset')
    ra_or_cl = rename(ra_or_cl, 'e_wm_t', 'e_wrap_m_trim')
    ra_or_cl = rename(ra_or_cl, 'ib_wrp_tr_m', 'e_wrap_m_trimmed')
    ra_or_cl = rename(ra_or_cl, 'e_wn', 'e_wrap_n')
    ra_or_cl = rename(ra_or_cl, 'e_wn_f', 'e_wrap_n_filt')
    ra_or_cl = rename(ra_or_cl, 'e_wn_t', 'e_wrap_n_trim')
    ra_or_cl = rename(ra_or_cl, 'ib_dm', 'ib_dyn_m')
    ra_or_cl = rename(ra_or_cl, 'ib_dn', 'ib_dyn_n')
    ra_or_cl = rename(ra_or_cl, 'ibm', 'ib_model')
    ra_or_cl = rename(ra_or_cl, 'ibmh', 'ib_amp_hdwe')
    ra_or_cl = rename(ra_or_cl, 'ibmh_f', 'ib_amp_hdwe_f')
    ra_or_cl = rename(ra_or_cl, 'ibmkf', 'ib_amp_hdwe_kf')
    ra_or_cl = rename(ra_or_cl, 'ibmm', 'ib_amp_model')
    ra_or_cl = rename(ra_or_cl, 'ibnh', 'ib_noa_hdwe')
    ra_or_cl = rename(ra_or_cl, 'ibnh_f', 'ib_noa_hdwe_f')
    ra_or_cl = rename(ra_or_cl, 'ibnkf', 'ib_noa_hdwe_kf')
    ra_or_cl = rename(ra_or_cl, 'ibnm', 'ib_noa_model')
    ra_or_cl = rename(ra_or_cl, 'vb_h', 'vb_hdwe')
    ra_or_cl = rename(ra_or_cl, 'vb_m', 'vb_model')

    return ra_or_cl


# Unix-like cat function
# e.g. > cat('out', ['in0', 'in1'], path_to_in='./')
def cat(out_file_name, in_file_names, in_path='./', out_path='./'):
    with open(os.path.join(out_path, out_file_name), 'w') as out_file:
        for in_file_name in in_file_names:
            with open(os.path.join(in_path, in_file_name)) as in_file:
                for line in in_file:
                    if line.strip():
                        out_file.write(line)
