import numpy as np


def aW_exp3(x):
    """
    Calculates growth rate aW based on movement speed
    Exponential model f(x) = a*e^(bx) + c
    a, b, c from scipy.optimize.curve_fit
    see test/plots.ipynb (5.)

    Parameters:
        x: movement speed

    Returns:
        aW
    """
    # return 0.07906346806320959*np.exp(-0.018379032171469217*x) + 0.06158365218448222 -> old parameters
    return 0.05598542781762466*np.exp(-0.02914930129288433*x) + 0.06196515212551092

def aW_exp2(x):
    """
    Calculates growth rate aW based on movement speed
    Exponential model f(x) = a*e^(bx)
    a, b from manual OLS
    see test/plots.ipynb (5.)

    Parameters:
        x: movement speed

    Returns:
        aW
    """
    return 0.116929*np.exp(-0.00455256*x)


def aW_lin(x):
    """
    Calculates growth rate aW based on movement speed
    Linear model based f(x) = mx+n
    see test/plots.ipynb (5.)

    Parameters:
        x: movement speed

    Returns:
        aW
    """
    return -0.000379166702*x + 0.1120833351
