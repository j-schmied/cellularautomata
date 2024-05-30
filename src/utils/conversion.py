import numpy as np
from numba import njit
from numba.core import types
from numba.typed import Dict


def convert_to_numba_dict(python_dict, unicode: bool = False):
    """
    Takes python dict and converts it to numba compatible dict.
    Types of keys and values must be uniform.

    Parameters:
        python_dict (dict)
        unicode (bool)

    Returns:
        numba_dict (numba.typed.Dict)
    """
    numba_dict = Dict.empty(key_type=types.unicode_type, value_type=(types.float64 if not unicode else types.unicode_type))

    for key, value in python_dict.items():
        numba_dict[key] = value

    return numba_dict


@njit()
def to_days(units, t):
    """Conversion of a monte carlo time step to days"""
    match(units["time"]):
        case "d":
            return t
        case "h":
            return t / 24
        case "min":
            return t / 1440
        case "s":
            return t / 86400
        case _:
            return -1


def time_conversion(t, tmax, dt, dc, plot_interval):
    """
    Coverts t-th time step from saved data to days based on dt

    Parameters:
        t (int)
        tmax (float)
        dt (float)
        dc (int)
        plot_interval (float)

    Returns:
        time conversion factor (float)
    """
    ts = np.array([x for x in range(0, int(tmax) + 1, int(dt)) if x % plot_interval == 0])
    interp = np.array(range(len(ts))) * dt * plot_interval

    factor = ts[-1] / interp[-1]
    interp *= factor

    return interp[t] / dc
