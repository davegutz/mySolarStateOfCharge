# GP_batteryEKF - general purpose battery class for EKF use
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

"""Define a general purpose battery model including Randles' model and SoC-VOV model as well as Kalman filtering
support for simplified Mathworks' tracker. See Huria, Ceraolo, Gazzarri, & Jackey, 2013 Simplified Extended Kalman
Filter Observer for SOC Estimation of Commercial Power-Oriented LFP Lithium Battery Cells.
Dependencies:
    - numpy      (everything)
    - matplotlib (plots)
    - reportlab  (figures, pdf)
"""
from json.encoder import encode_basestring

from unite_pictures import unite_pictures_into_pdf, cleanup_fig_files
import os
import ast


def extract(function_call_string: str) -> list[str]:
    """
    Reads a line of text that is a function call and produces a list
    containing the original text of its arguments.

    Args:
        function_call_string: A string containing a single Python function call.

    Returns:
        A list of strings, where each string is the text of an argument.
    """
    # ast.parse needs the input to be a valid expression.
    # The mode='eval' expects a single expression.
    try:
        # Remove leading/trailing whitespace for clean parsing
        source_line = function_call_string.strip()
        tree = ast.parse(source_line, mode='eval')
    except SyntaxError as e:
        print(f"ValueError source_line:  {source_line}")
        raise ValueError(f"Invalid function call syntax: {e}")

    # The body of the expression should be an ast.Call node
    call_node = tree.body
    if not isinstance(call_node, ast.Call):
        raise ValueError("Provided string is not a function call expression")

    arguments_text = []

    # 1. Process positional arguments
    for arg in call_node.args:
        # ast.get_source_segment extracts the original source code for the node
        arg_text = ast.get_source_segment(source_line, arg)
        if arg_text is not None:
            arguments_text.append(arg_text)

    # 2. Process keyword arguments (e.g., name="value")
    for kwarg in call_node.keywords:
        key_text = kwarg.arg  # The keyword name (string)
        # Get the source segment for the value part
        value_text = ast.get_source_segment(source_line, kwarg.value)
        if value_text is not None:
            arguments_text.append(f"{key_text}={value_text}")

    return arguments_text


class Arg:

    def __init__(self, directive=None, value=None):
        self.directive = directive  # e.g. 'add='
        self.val = value
        
    def __str__(self):
        if self.directive is None and self.val is None:
            return ''
        else:
            return ', ' + self.directive + self.val


class Line:
    def __init__(self, in_str):
        self.header = in_str.split('(')[0]
        in_list = extract(in_str)
        self.plt_dir = 'plt'
        self.x = None
        self.x_txt = None
        self.y = None
        self.y_text = None
        self.add_arg = Arg()
        self.slr_arg = Arg()
        self.col_arg = Arg()
        self.ls_arg = Arg()
        self.mk_arg = Arg()
        self.mk_sz_arg = Arg()
        self.mk_ev_arg= Arg()
        self.stairs_arg = Arg()
        self.warn_arg = Arg()
        self.lw_arg = Arg()
        i = -1
        for I in in_list:
            i += 1
            if i == 1:
                self.x = I
            elif i == 2:
                self.x_txt = I
            elif i == 3:
                self.y = I
            elif i == 4:
                self.y_txt = I
            if I.__contains__('label='):  # labels are built into plq if label=None.  So skip them
                continue
            elif I.__contains__('add='):
                self.add_arg = Arg('add=', I.replace('add=', ""))
            elif I.__contains__('slr='):
                self.slr_arg = Arg('slr=', I.replace('slr=', ""))
            elif I.__contains__('color='):
                self.col_arg = Arg('color=', I.replace('color=', ""))
            elif I.__contains__('linestyle='):
                self.ls_arg = Arg('linestyle=', I.replace('linestyle=', ""))
            elif I.__contains__('linewidth='):
                self.ls_arg = Arg('linewidth=', I.replace('linewidth=', ""))
            elif I.__contains__('marker='):
                self.mk_arg = Arg('marker=', I.replace('marker=', ""))
            elif I.__contains__('markersize='):
                self.mk_sz_arg = Arg('markersize=', I.replace('markersize=', ""))
            elif I.__contains__('markevery='):
                self.mk_ev_arg = Arg('markevery=', I.replace('markevery=', ""))
            elif I.__contains__('warn='):
                self.warn_arg = Arg('warn=', I.replace('warn=', ""))
            elif I.__contains__('stairs='):
                self.stairs_arg = Arg('stairs=', I.replace('stairs=', ""))

    def __str__(self):
        ostr = self.header + '('
        ostr += self.plt_dir
        ostr += ', ' + self.x
        ostr += ', ' + self.x_txt
        ostr += ', ' + self.y
        ostr += ', ' + self.y_txt
        ostr += self.add_arg.__str__()
        ostr += self.slr_arg.__str__()
        ostr += self.col_arg.__str__()
        ostr += self.ls_arg.__str__()
        ostr += self.lw_arg.__str__()
        ostr += self.mk_arg.__str__()
        ostr += self.mk_sz_arg.__str__()
        ostr += self.mk_ev_arg.__str__()
        ostr += self.stairs_arg.__str__()
        ostr += self.warn_arg.__str__()
        ostr += ")\n"
        if ostr.count("color=") > 1:
            pass
        return ostr


def do_one(path_to_infile, path_to_outfile):
    num_plq_in = 0
    os.remove(path_to_outfile)
    print(f"doing {path_to_infile} --> {path_to_outfile}")
    with (open(path_to_infile, "r", encoding='cp437') as input_file):  # reads all characters even bad ones
        with open(path_to_outfile, "a") as output:
            lines = input_file.readlines()
            i = 0
            while i < len(lines):
                line = lines[i]
                if "plq(" in line and not "# " in line and not "def" in line:
                    num_plq_in += 1
                    line_ends_comma = (line.count(",\n") > 0) or (line.count(", \n") > 0)
                    # print(f" comma? {line_ends_comma}  {line}")
                    if line_ends_comma:
                        next_line = lines[i+1]
                        combined_line = line + " " + next_line.strip() + "\n"
                        line = combined_line
                        # print(f"Error fixed: {combined_line}")
                        i += 1
                    trans = Line(line)
                    output.write(trans.__str__())
                else:
                    output.write(line)
                i += 1

def main():
    do_one('./CompareFault - Copy.py', './CompareFault.py')
    do_one('./DataOverModel - Copy.py', './DataOverModel.py')
    do_one('./PlotEKF - Copy.py', './PlotEKF.py')
    do_one('./PlotHist - Copy.py', './PlotHist.py')
    do_one('./CompareHistSim - Copy.py', './CompareHistSim.py')
    do_one('./PlotSimS - Copy.py', './PlotSimS.py')
    do_one('./PlotGP - Copy.py', './PlotGP.py')
    do_one('./PlotOffOn - Copy.py', './PlotOffOn.py')
    do_one('./Battery - Copy.py', './Battery.py')


if __name__ == "__main__":
    import sys
    main()
