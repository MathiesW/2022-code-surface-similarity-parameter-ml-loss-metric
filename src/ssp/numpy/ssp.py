import numpy as np


def sobolev_norm(y: np.ndarray) -> np.ndarray:
    return np.linalg.norm(y, axis=-1)


def ssp(y1: np.ndarray, y2: np.ndarray, batched: bool = False) -> np.ndarray:
    """ basic SSP implementation for discrete signals, no lowpass etc.

    # args
        y1: np.ndarray
        y2: np.ndarray
    """

    if batched:
        # flatten inputs along last axis to keep batch-dimension, batch dimension is maintained
        y1 = np.reshape(y1, (y1.shape[0], -1))
        y2 = np.reshape(y2, (y2.shape[0], -1))
    else:
        # flatten inputs completely, result will be a float
        y1 = np.ravel(y1)
        y2 = np.ravel(y2)

    # get numerator and denuminator
    numerator = sobolev_norm(y1 - y2)
    denum = sobolev_norm(y1) + sobolev_norm(y2)

    return np.divide(
        numerator,
        denum,
        where=denum != 0
    )
