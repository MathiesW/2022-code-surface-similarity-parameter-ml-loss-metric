import numpy as np


def sobolev_norm(y):
    """
    Sobolev / L2 norm along last axis by means of np.linalg.norm

    Parameters
    ----------
    y : array_like
        Input array. Must be either 1-D or 2-D

    Returns
    -------
    n : float or ndarray
        Norm of the matrix or vector(s).
    """
    return np.linalg.norm(y, axis=-1)


def ssp(y1, y2, batched=False):
    """
    Surface Similarity Parameter (SSP) without any fuzz, cf. https://doi.org/10.1007/s10665-016-9849-7

    Parameters
    ----------
    y1 : array_like
    y2 : array_like
    batched : bool, optional
        If this is set to True, the result is not reduced along the first dimension.
        This option allows to use the SSP for comparing batches of data.

    Returns
    -------
    y : float or ndarray
        Surface Similarity Parameter between arrays y1 and y2.

    Notes
    -----
    The result is always within [0, 1] with
        0 indicating perfect agreement, and
        1 indicating perfect disagreement.
    'Perfect disagreement' in terms of SSP means either
        - a non-zero signal is compared against a zero signal, or
        - a signal is compared against the inverse itself, e.g., ssp(x, -x)

    Examples
    --------
    >>> from ssp.numpy import ssp
    >>> import numpy as np
    >>> np.random.seed(0)  # for deterministic results
    >>> y1 = np.random.random((2, 32))
    >>> y2 = np.random.random((2, 32))
    >>> ssp(y1, y2)  # some value between 0 and 1
    np.float64(0.35119514129237195)
    >>> ssp(y1, y1)
    np.float64(0.0)
    >>> ssp(y1, -y1)
    np.float64(1.0)
    >>> ssp(y1, np.zeros_like(y1))
    np.float64(1.0)
    
    Use 'batched=True' to get a unique result for each signal in y1 and y2

    >>> from ssp.numpy import ssp
    >>> import numpy as np
    >>> np.random.seed(0)  # for deterministic results
    >>> y1 = np.random.random((2, 32))
    >>> y2 = np.random.random((2, 32))
    >>> ssp(y1, y2, batched=True)
    array([0.34864963, 0.35827101])
    >>> ssp(y1, y1, batched=True)
    array([0., 0.])
    >>> ssp(y1, -y1, batched=True)
    array([1., 1.])
    >>> ssp(y1, np.zeros_like(y1), batched=True)
    array([1., 1.])
    
    """
    
    assert y1.shape == y2.shape, f"Shapes have to match, received {y1.shape} and {y2.shape}"
    
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
