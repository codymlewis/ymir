"""
General function-related utility functions.
"""


def chain(funclist, x):
    """Chain a list of function together and return their composition upon x"""
    for f in funclist:
        x = f(x)
    return x