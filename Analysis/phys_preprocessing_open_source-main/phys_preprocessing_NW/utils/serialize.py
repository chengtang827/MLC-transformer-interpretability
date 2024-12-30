"""Function to serialize a variable.

This is useful to apply to a variable before converting to json format for
saving.
"""

import numpy as np


def serialize(x):
    """Serialize an input x."""
    if isinstance(x, np.int_):
        x = int(x)
    elif isinstance(x, np.float_):
        x = float(x)
    elif isinstance(x, np.ndarray):
        x = x.tolist()
    elif isinstance(x, dict):
        x = {k: serialize(v) for k, v in x.items()}
    return x
