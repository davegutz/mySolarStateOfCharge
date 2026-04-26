import glob
import os
from pathlib import PurePosixPath
import re

from reportlab.lib import utils
from reportlab.pdfgen import canvas


def convert(text):
    return int(text) if text.isdigit() else text


def alphanum_key(key):
    return [convert(c) for c in re.split('([0-9]+)', key)]


def sorted_nicely(_l):
    # noinspection HttpUrlsUsage
    """
        # http://stackoverflow.com/questions/2669059/how-to-sort-alpha-numeric-set-in-python

        Sort the given iterable in the way that humans expect.
        """
    return sorted(_l, key=alphanum_key)


def cleanup_fig_files(fig_files):
    # Clean up after itself.   Other fig files already in root will get plotted by unite_pictures_into_PDF
    # Cleanup other figures in root folder by hand
    for fig_file in fig_files:
        try:
            os.remove(fig_file)
        except OSError:
            pass


# noinspection GrazieInspection
def precleanup_fig_files(output_pdf_name='unite_pictures.pdf', path_to_pdfs='.'):
    # Clean up before itself.   Other fig files already in root will get plotted by unite_pictures_into_pdf
    # Cleanup other figures in root folder by hand
    from glob import glob
    from os import remove
    for file in glob(str(PurePosixPath(path_to_pdfs) / (output_pdf_name + '*.pdf'))):
        print("\nremoving", file, end='')
        try:
            remove(file)
        except OSError:
            pass


# ----------------------------------------------------------------------
def pngs_to_pdf(png_folder='.', output_pdf='output.pdf'):
    """Combine all PNGs in png_folder into a single PDF at output_PDF (full path)."""
    pngs = sorted_nicely(glob.glob(str(PurePosixPath(png_folder) / '*.png')))
    if not pngs:
        print("pngs_to_pdf: no PNG files found in", png_folder)
        return
    c = canvas.Canvas(output_pdf)
    c.setTitle(PurePosixPath(output_pdf).stem)
    for png in pngs:
        img = utils.ImageReader(png)
        w, h = img.getSize()
        c.setPageSize((w, h))
        c.drawImage(png, 0, 0)
        c.showPage()
    c.save()
    print("created", output_pdf)
