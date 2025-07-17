import numpy as np


def ssp(x1, x2, batched=False):
    """
    Surface Similarity Parameter (SSP) without any fuzz, cf. https://doi.org/10.1007/s10665-016-9849-7.
    This implementation does not rely in any FFT call and thus complies with n-D data.

    Parameters
    ----------
    x1 : ndarray
        First input array for comparison
    x2 : ndarray
        Second input array for comparison
    batched : bool, optional
        If this is set to `True`, the result is not reduced along the first dimension.
        This option allows to use the SSP for comparing batches of data.

    Returns
    -------
    ssp : ndarray
        Surface Similarity Parameter between arrays x1 and x2.

    Notes
    -----
    The result is always within [0, 1] with
        - 0 indicating perfect agreement, and
        - 1 indicating perfect disagreement.
    `Perfect disagreement` in terms of SSP means either
        - a non-zero signal is compared against a zero signal, or
        - a signal is compared against the inverse itself, e.g., `ssp(x, -x)`.

    Examples
    --------
    >>> from ssp.numpy import ssp
    >>> import numpy as np
    >>> np.random.seed(0)  # for deterministic results
    >>> x1 = np.random.random((2, 32))
    >>> x2 = np.random.random((2, 32))
    >>> ssp(x1, x2)  # some value between 0 and 1
    np.float64(0.35119514129237195)
    >>> ssp(x1, x1)
    np.float64(0.0)
    >>> ssp(x1, -x1)
    np.float64(1.0)
    >>> ssp(x1, np.zeros_like(x1))
    np.float64(1.0)
    
    Use `batched=True` to get a unique result for each signal in `x1` and `x2`

    >>> from ssp.numpy import ssp
    >>> import numpy as np
    >>> np.random.seed(0)  # for deterministic results
    >>> x1 = np.random.random((2, 32))
    >>> x2 = np.random.random((2, 32))
    >>> ssp(x1, x2, batched=True)
    array([0.34864963, 0.35827101])
    >>> ssp(x1, x1, batched=True)
    array([0., 0.])
    >>> ssp(x1, -x1, batched=True)
    array([1., 1.])
    >>> ssp(x1, np.zeros_like(x1), batched=True)
    array([1., 1.])
    
    """
    
    assert x1.shape == x2.shape, f"Shapes have to match, received {x1.shape} and {x2.shape}"
    
    if batched:
        # flatten inputs along last axis to keep batch-dimension, batch dimension is maintained
        x1 = np.reshape(x1, (x1.shape[0], -1))
        x2 = np.reshape(x2, (x2.shape[0], -1))
    else:
        # flatten inputs completely, result will be a float
        x1 = np.ravel(x1)
        x2 = np.ravel(x2)

    # get numerator and denuminator
    numerator = np.linalg.norm(x1 - x2, axis=-1)
    denum = np.linalg.norm(x1, axis=-1) + np.linalg.norm(x2, axis=-1)

    return np.divide(
        numerator,
        denum,
        where=denum != 0
    )
