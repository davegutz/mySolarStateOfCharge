# Battery - general purpose battery class for modeling
# Copyright (C) 2026 Dave Gutz
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation;
# version 2.1 of the License.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# See http://www.fsf.org/licensing/licenses/lgpl.txt for full license text.

# type: ignore
# noinspection PyPep8Naming,PyUnresolvedReferences,PyAttributeOutsideInit,PyArgumentList,PyCallingNonCallable,PyUnboundLocalVariable,PyUnfilledParameters
# pylint: disable=invalid-name, no-member, attribute-defined-outside-init, used-before-assignment, redefined-outer-name, redefined-builtin

"""Depending on platform locate the temp folders."""

import os
import platform
from pathlib import Path, PurePosixPath


def local_paths(version_folder='no_name'):
    if platform.system() == 'Linux':
        path_to_local = '/home/daveg/.local/SOC_Particle'
    elif platform.system() == 'Darwin':
        path_to_local = '/Users/daveg/.local/SOC_Particle'
    else:
        path_to_local = str(Path(os.getenv('LOCALAPPDATA') or '.') / 'SOC_Particle')

    if not Path(path_to_local).is_dir():
        os.mkdir(path_to_local)
    path_to_version_folder = str(PurePosixPath(path_to_local) / version_folder)
    if not Path(path_to_version_folder).is_dir():
        os.mkdir(path_to_version_folder)
    path_to_temp = str(PurePosixPath(path_to_version_folder) / 'temp')
    if not Path(path_to_temp).is_dir():
        os.mkdir(path_to_temp)
    save_pdf_path = str(PurePosixPath(path_to_version_folder) / 'figures')
    if not Path(save_pdf_path).is_dir():
        os.mkdir(save_pdf_path)
    putty_test_csv_path = str(PurePosixPath(path_to_local) / 'putty_test.csv')

    return path_to_temp, save_pdf_path, putty_test_csv_path


def version_from_data_file(path_to_data):
    data_file_folder = str(PurePosixPath(path_to_data).parent)
    version = PurePosixPath(data_file_folder).name
    return version


def version_from_data_path(path):
    version = PurePosixPath(path).name
    return version
