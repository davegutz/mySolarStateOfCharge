# Scale class transition
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

__author__ = 'Dave Gutz <davegutz@alum.mit.edu>'
__version__ = '$Revision: 1.1 $'
__date__ = '$Date: 2025/09/27 13:15:02 $'


class Scale:
    # Scale output between two values as input varies between two values.

    def __init__(self, i_n, i_x, o_n, o_x):
        # Defaults
        self.i_n = i_n
        self.i_x = i_x
        self.o_n = o_n
        self.o_x = o_x
        self.di = i_x - i_n
        self.do = o_x - o_n

    def calculate(self, i):
        if i <= self.i_n:
            o = self.o_n
        elif i >= self.i_x:
            o = self.o_x
        else:
            o = (i - self.i_n) / self.di * self.do + self.o_n
        return o * o

    def __str__(self, prefix=''):
        """Returns representation of the object"""
        s = prefix + "Scale:\n"
        s += "  i_n = {:7.3f}    // Minimum input break\n".format(self.i_n)
        s += "  i_x = {:7.3f}    // Maximum input break\n".format(self.i_x)
        s += "  o_n = {:7.3f}    // Minimum output break\n".format(self.o_n)
        s += "  o_x = {:7.3f}    // Maximum output break\n".format(self.o_x)
        s += "  di =  {:7.3f}    // Delta input breaks\n".format(self.di)
        s += "  do =  {:7.3f}    // Delta output breaks\n".format(self.do)
        return s


class ScaleSelector:
    """ Scale select between a high and low set of inputs.  Low might be a precise, amplified sensor and high might be the high range equivalent
    """

    def __init__(self, n_lo, n_hi, p_lo, p_hi):
        # Defaults
        self.n_lo = n_lo
        self.n_hi = n_hi
        self.p_lo = p_lo
        self.p_hi = p_hi
        self.n_d = n_hi - n_lo
        self.p_d = p_hi - p_lo

    """
                      ^ scale
                      |
    ------            |          ------- 1.0 ==> all lg
           -          |        -
             -        |      -
          |    -------------             0.0 ==> all sm
          |    |      |     |    |
       n_lo   n_hi    |   p_lo   p_hi
                      |
                      |
                      v
    n_d = n_hi - n_lo
    p_d = p_hi - p_lo
    
    """
    def scale_select(self, inp, sm, lg):
        if self.n_hi <= inp <= self.p_lo:
            return sm
        elif inp <= self.n_lo or inp >= self.p_hi:
            return lg
        elif inp < self.n_hi:
            return (inp - self.n_lo) / self.n_d * ( sm - lg ) + lg
        else:
            return (inp - self.p_lo) / self.p_d * ( lg - sm ) + sm

    def __str__(self, prefix=''):
        """Returns representation of the object"""
        s = prefix + "Scale:\n"
        s += "  i_big_neg = {:7.3f}    // Negative big break\n".format(self.i_big_neg)
        s += "  i_small_neg = {:7.3f} // Negative small break\n".format(self.i_small_neg)
        s += "  i_small_pos = {:7.3f} // Positive big break\n".format(self.i_small_pos)
        s += "  del_neg = {:7.3f}     // Negative break span, signed\n".format(self.del_neg)
        s += "  del_pos = {:7.3f}     // Positive break span, signed\n".format(self.del_pos)
        return s
