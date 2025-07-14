from keras import ops
from keras import KerasTensor
from math import pi


def hard_lowpass(n: int, spectral_radius: int, truly_symmetric: bool = False):
    """
    1D hard lowpass
    Args:
        n: grid size of data
        spectral_radius: width of lowpass
        truly_symmetric (bool): if True, ops.linspace is used insted of ops.arange
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


def circular_hard_lowpass(n: int, spectral_radius: int, truly_symmetric: bool = False):
    """
    2D hard lowpass
    Args:
        n: grid size of data
        spectral_radius: width of lowpass
        truly_symmetric (bool): if True, ops.linspace is used insted of ops.arange
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


def fftfreq(n: int, d: float, rad: bool = True) -> KerasTensor:
    fs = 1 / d
    df = fs / n
    fft_freqs = ops.arange(-(n // 2) * df, (n // 2 * df), df)

    if rad:
        fft_freqs *= (2 * pi)

    return ops.roll(fft_freqs, shift=n // 2)


def squeeze_or_expand_to_same_rank(x1: KerasTensor, x2: KerasTensor, axis: int = -1, expand_rank_1: bool = True) -> tuple:
    """Squeeze/expand last dim if ranks differ from expected by exactly 1."""
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
