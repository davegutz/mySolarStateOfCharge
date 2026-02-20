import matplotlib.pyplot as plt
plt.rcParams['axes.grid'] = True
plt.rcParams['legend.fontsize'] = 'small'
plt.rcParams['savefig.dpi'] = 300  # Set default savefig DPI to 300
# plt.rcParams['figure.dpi'] = 300  # Also increase display DPI for consistency

# Monkey-patch savefig: resize to 16:9 for the save, restore display size after.
_orig_savefig = plt.savefig

def _savefig_fullscreen(*args, **kwargs):
    fig = plt.gcf()
    orig_size = fig.get_size_inches()
    fig.set_size_inches(16, 9)
    _orig_savefig(*args, **kwargs)
    fig.set_size_inches(*orig_size)

plt.savefig = _savefig_fullscreen
