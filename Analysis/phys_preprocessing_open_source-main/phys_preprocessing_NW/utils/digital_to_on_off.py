"""Function to convert digital vector to on/off indices."""

from matplotlib import pyplot as plt
import numpy as np


def digital_to_on_off(digital, plot=False):
    """Convert digital vector to on/off indices.
    
    Args:
        digital: 1-dimensional binary lost or numpy array.
        plot: Bool. If True, plot digital and threshold convolved with digital.
    
    Returns:
        on: Indices where digital goes from 0 to 1.
        off: Indices where digital goes from 1 to 0.
    """

    # Check that input is digital
    non_digital = [x for x in digital if x not in [0, 1]]
    if len(non_digital):
        raise ValueError(
            'Input is not digital. First non-digital elements are '
            f'{non_digital[:100]}'
        )

    # Compute switches by convolving with [1, -1]
    digital = np.array(digital)
    conv = np.convolve(2 * digital - 1., np.array([1, -1]))
    on = np.argwhere(conv == 2)[:, 0]
    off = np.argwhere(conv == -2)[:, 0]

    # Plot if necessary
    if plot:
        _, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].plot(digital)
        axes[1].plot(conv)
    
    return on, off