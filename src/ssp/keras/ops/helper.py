from keras import ops
from keras import KerasTensor
from math import pi


def hard_lowpass(n, spectral_radius, truly_symmetric=False):
    """
    1-D binary mask of shape `n` that can be multiplied with the FFT frequencies ::

      hard_lowpass = [0, ..., 0, 1, ..., 1, 0, ..., 0]

    Parameters
    ----------
    n : int
        Grid size of the data
    spectral_radius : int
        Width of the lowpass
    truly_symmetric : bool, optional
        If this is set, ops.linspace is used insted of ops.arange to achieve a truly symmetric filter (for odd `n`).
        Defaults to `False`.

    Returns
    -------
    hard_lowpass : KerasTensor
        Binary 1-D mask of size `(n,)`

    """

    if truly_symmetric:
        x = ops.linspace(-n//2, n//2, n)
    else:
        x = ops.arange(n) - n//2

    x = ops.cast(x, dtype="float32")

    hard_lowpass = ops.where(
        ops.greater_equal(ops.abs(x), ops.cast(spectral_radius, dtype="float32")),
        0.0,
        1.0
    )
    return hard_lowpass


def circular_hard_lowpass(n, spectral_radius, truly_symmetric=False):
    """
    2-D binary mask of shape `(n,n)` that can be multiplied with the FFT frequencies, e.g., ::

      circular_hard_lowpass = [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0]
      ]

    Parameters
    ----------
    n : int
        Grid size of the data, square grid `(n,n)`
    spectral_radius : int
        Radius of the lowpass on the `(n,n)` grid
    truly_symmetric : bool, optional
        If this is set, ops.linspace is used insted of ops.arange to achieve a truly symmetric filter (for odd `n`).
        Defaults to `False`.

    Returns
    -------
    hard_lowpass : KerasTensor
        Binary 2-D mask of size `(n,n)`

    """

    if truly_symmetric:
        x = ops.linspace(-n//2, n//2, n)
    else:
        x = ops.arange(n) - n//2

    grid = ops.sqrt(ops.sum(ops.square(ops.meshgrid(x, x)), axis=0))

    circular_hard_lowpass = ops.where(
        ops.greater_equal(grid, ops.cast(spectral_radius, dtype="float32")),
        0.0,
        1.0
    )
    return circular_hard_lowpass


def fftfreq(n, d=1, rad=False):
    """
    Return the Discrete Fourier Transform sample frequencies.

    The returned float array `f` contains the frequency bin centers in cycles
    per unit of the sample spacing (with zero at the start).  For instance, if
    the sample spacing is in seconds, then the frequency unit is cycles/second.

    Given a window length `n` and a sample spacing `d`::

      f = [0, 1, ...,   n/2-1,     -n/2, ..., -1] / (d*n)   if n is even
      f = [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] / (d*n)   if n is odd

    Parameters
    ----------
    n : int
        Window length.
    d : scalar, optional
        Sample spacing (inverse of the sampling rate). Defaults to `1`.
    rad : bool, optional
        If this is set, the angular frequency `omega=2*pi*f` is returned.
        Defaults to `False`.

    Returns
    -------
    f : KerasTensor
        Tensor of length `n` containing the sample frequencies.

    Examples
    --------
    >>> from keras import ops
    >>> from ssp.keras.ops import fft, fftfreq
    >>> signal = ops.array([-2, 8, 6, 4, 1, 0, 3, 5], dtype=float)
    >>> fourier = fft(signal)
    >>> n = ops.size(signal)
    >>> timestep = 0.1
    >>> freq = fftfreq(n, d=timestep)
    >>> freq
    array([ 0.  ,  1.25,  2.5 , ..., -3.75, -2.5 , -1.25])

    """

    fs = 1.0 / d
    df = fs / ops.cast(n, float)
    fft_freqs = ops.arange(-ops.cast(n // 2, float) * df, ops.cast(n // 2, float) * df, df)

    if rad:
        fft_freqs *= (2 * pi)

    return ops.roll(fft_freqs, shift=n // 2)


def squeeze_or_expand_to_same_rank(x1, x2, axis=-1, expand_rank_1: bool = True) -> tuple:
    """
    Squeeze/expand along `axis` if ranks differ from expected by exactly 1.

    Parameters
    ----------
    x1 : KerasTensor
        first input tensor
    x2 : KerasTensor
        second input tensor
    axis : int, optional
        axis to squeeze or expand along. Defaults to `-1`.
    expand_rank_1: bool, optional
        Defaults to `True`

    Returns
    -------
    x1, x2 : (KerasTensor, KerasTensor)
        Tuple of `(x1, x2)` with the same shape

    """

    x1_rank = len(x1.shape)
    x2_rank = len(x2.shape)
    if x1_rank == x2_rank:
        return x1, x2
    if x1_rank == x2_rank + 1:
        if x1.shape[axis] == 1:
            if x2_rank == 1 and expand_rank_1:
                x2 = ops.expand_dims(x2, axis=axis)
            else:
                x1 = ops.squeeze(x1, axis=axis)
    if x2_rank == x1_rank + 1:
        if x2.shape[axis] == 1:
            if x1_rank == 1 and expand_rank_1:
                x1 = ops.expand_dims(x1, axis=axis)
            else:
                x2 = ops.squeeze(x2, axis=axis)
    return x1, x2
