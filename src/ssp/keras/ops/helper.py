import inspect
from keras import ops
from keras import layers
from keras import saving
from keras import KerasTensor
from keras.src import backend
import string
from math import pi
import re


def tukey(n: int, m: int = None, alpha: float = 0.25, truly_symmetric: bool = False) -> KerasTensor:
    """
    1D tukey window
    Args:
        n: grid size of data
        m: width of tukey window
        alpha: width of slope
        truly_symmetric (bool): if True, ops.linspace is used insted of ops.arange
    """
    if truly_symmetric:
        x = ops.linspace(-n//2, n//2, n)
    else:
        x = ops.arange(n) - n//2

    x = ops.cast(x, dtype="float32")
    d = ops.cast(ops.minimum(n, 2*m), dtype='float32')
    d = ops.maximum(d, 64)  # set lower bound of d to 64

    width = ops.cast(ops.floor(alpha * (d - 1.0) / 2.0), dtype='int32')  # the width of the ramp!

    tukey = ops.where(
        ops.greater_equal(ops.abs(x), (d // 2.0 - ops.cast(width, dtype='float32'))),
        ops.where(# here comes the cosine decay
            ops.less(ops.abs(x), d // 2.0),
            0.5 * (1 + ops.cos(pi * (2.0 * x / alpha / (d - 1.0) - 1.0))),
            0.0
        ),  
        1.0  # center is always 1.0
    )

    return tukey


def circular_tukey(n: int, m: int = None, alpha: float = 0.25, truly_symmetric: bool = False) -> KerasTensor:
    """
    2D tukey window with a circular support
    Args
        n: grid size of data
        m: radius of tukey window
        alpha: width of slope
        truly_symmetric (bool): if True, ops.linspace is used insted of ops.arange 
    """
    if truly_symmetric:
        x = ops.linspace(-n//2, n//2, n)
    else:
        x = ops.arange(n) - n//2

    grid = ops.sqrt(ops.sum(ops.square(ops.meshgrid(x, x)), axis=0))
    d = ops.cast(ops.minimum(n, 2*m), dtype='float32')
    d = ops.maximum(d, 64)  # set lower bound of d to 64

    width = ops.cast(ops.floor(alpha * (d - 1.0) / 2.0), dtype='int32')  # the width of the ramp!

    circular_tukey = ops.where(
        ops.greater_equal(grid, (d // 2.0 - ops.cast(width, dtype='float32'))),
        ops.where(# here comes the cosine decay
            ops.less(grid, d // 2.0),
            0.5 * (1 + ops.cos(pi * (2.0 * grid / alpha / (d - 1.0) - 1.0))),
            0.0
        ),  
        1.0  # center is always 1.0
    )

    return circular_tukey


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


def trapz(y: KerasTensor, axis: int = -1) -> KerasTensor:
    nd = ops.ndim(y)

    slice1 = [slice(None)] * nd
    slice2 = [slice(None)] * nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    
    return ops.sum((y[tuple(slice1)] + y[tuple(slice2)]) / 2.0, axis=axis, keepdims=False)


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


def large_negative_number(dtype):
    """Return a Large negative number based on dtype."""
    if backend.standardize_dtype(dtype) == "float16":
        return -3e4
    return -1e9


def index_to_einsum_variable(i):
    """Coverts an index to a einsum variable name.

    We simply map indices to lowercase characters, e.g. 0 -> 'a', 1 -> 'b'.
    """
    return string.ascii_lowercase[i]


def unwrap(phase: KerasTensor, axis=-1) -> KerasTensor:
    nd = ops.ndim(phase)

    # get phase difference and correction
    phase_diff = ops.diff(phase, axis=axis)
    jumps = ops.cast(phase_diff < -pi, dtype="int8") - ops.cast(phase_diff > pi, dtype="int8")
    correction = ops.cumsum(ops.cast(jumps, dtype="float32") * 2.0 * pi, axis=axis)

    # pad to original size
    pad_width = [(0, 0)] * nd
    pad_width[axis] = (1, 0)

    correction = ops.pad(correction, pad_width=tuple(pad_width))
    return phase + correction


def capitalize_first_char(s: str):
    """Capitalize first character of string and leave the rest
    source: https://stackoverflow.com/questions/12410242/python-capitalize-first-letter-only

    use to import layers, e.g., "Conv2D".capitalize() results in "Conv2d", which is no valid layer!
    """
    return re.sub('([a-zA-Z])', lambda x: x.groups()[0].upper(), s, 1)


@saving.register_keras_serializable(package="Helper", name="get_layer")
def get_layer(identifier, module: str = "keras.layers", registered_name: str = None, **layer_kwargs):
    if identifier is None:
        return None
    if isinstance(identifier, dict):
        obj = layers.deserialize(identifier)
    elif isinstance(identifier, str):
        config = {
            "module": module,
            "class_name": str(capitalize_first_char(identifier)),  # layer names are all capital!
            "config": {
                "name": None,
                "trainable": True,
                "dtype": {
                    "module": "keras",
                    "class_name": "DTypePolicy",
                    "config": {"name": "float32"},
                    "registered_name": None
                },
                **layer_kwargs
            },
            "registered_name": registered_name
        }
        obj = layers.deserialize(config)
    else:
        obj = identifier

    if callable(obj):
        if inspect.isclass(obj):
            obj = obj()
        return obj
    else:
        raise ValueError(
            f"Could not interpret layer identifier: {identifier}"
        )