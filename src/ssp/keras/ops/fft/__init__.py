from keras.src.backend.config import backend
from keras.src.backend import any_symbolic_tensors
from keras.src.ops.operation import Operation
from keras import ops
from keras import KerasTensor
from typing import Union, Tuple

if backend() == 'jax':
    from .jax import fft_fn, ifft_fn, fft2_fn, ifft2_fn, rfft_fn, irfft_fn, rfft2_fn, irfft2_fn

if backend() == 'tensorflow':
    from .tensorflow import fft_fn, ifft_fn, fft2_fn, ifft2_fn, rfft_fn, irfft_fn, rfft2_fn, irfft2_fn


def cast_to_complex(x: Union[Tuple[KerasTensor, KerasTensor], KerasTensor]) -> Tuple[KerasTensor, KerasTensor]:
    if isinstance(x, tuple):
        return x
    else:
        return x, ops.zeros_like(x, dtype=x.dtype)


# === FFT ===
class FFT(Operation):
    """
    Base 1-D fast Fourier transform (FFT) operation.
    Keras-backend-agnostic version of the FFT.

    Notes
    -----
    Keras3 does not support complex dtypes.
    Therefore, the `keras.ops.fft` is quite cumbersume to use, since it explicitly requires the definition of a real and a complex part when calling `fft` ::

        >>> from keras import ops
        >>> x = ops.ones((4,))
        >>> ops.fft(x)
        ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
        >>> ops.fft((x, ops.zeros_like(x)))
        array([4.,  0.,  0.,  0.], dtype=float32), array([0., 0., 0., 0.], dtype=float32)
    
    This implementation attempts to increase the user-friendliness of the FFT call in Keras by handling the input automatically.

    """

    def __init__(self):
        super().__init__()
        self.fft_fn = fft_fn

    def compute_output_spec(self, x):
        """
        Compute output spec of Fourier transform

        Parameters
        ----------
        x : KerasTensor | tuple | list
            Real- or complex input to FFT. A complex input must be composed of a tuple or list of the real- and imaginary part `(x_real, x_imag)`.

        Returns
        -------
        y_real_spec, y_imag_spec : (KerasTensor, KerasTensor)
            spec of real- and imaginary part of `FFT(x)`

        """

        if not isinstance(x, (tuple, list)) or len(x) != 2:
            real = x
            imag = ops.zeros_like(x)
        else:
            real, imag = x
        # Both real and imaginary parts should have the same shape.
        if real.shape != imag.shape:
            raise ValueError(
                "Input `x` should be a tuple of two tensors - real and "
                "imaginary. Both the real and imaginary parts should have the "
                f"same shape. Received: x[0].shape = {real.shape}, "
                f"x[1].shape = {imag.shape}"
            )

        return (
            KerasTensor(shape=real.shape, dtype=real.dtype),
            KerasTensor(shape=imag.shape, dtype=imag.dtype),
        )

    def call(self, x):
        """
        Call method of FFT

        Parameters
        ----------
        x : KerasTensor | tuple | list
            Real- or complex input to FFT. A complex input must be composed of a tuple or list of the real- and imaginary part `(x_real, x_imag)`.

        Returns
        -------
        y_real, y_imag : (KerasTensor, KerasTensor)
            real- and imaginary part of `FFT(x)`

        Notes
        -----
        The input can be either a KerasTensor or a tuple or list of length 2, containing the real- and imaginary part of `x`.

        """

        return self.fft_fn(x)
    

class FFT2(FFT):
    """
    2-D fast Fourier transform

    """

    def __init__(self):
        super.__init__()
        self.fft_fn = fft2_fn


class RFFT(FFT):
    """
    1-D fast Fourier transform for real-valued inputs

    """

    def __init__(self):
        super.__init__()
        self.fft_fn = rfft_fn


class RFFT2(FFT):
    """
    2-D fast Fourier transform for real-valued inputs

    """

    def __init__(self):
        super.__init__()
        self.fft_fn = rfft2_fn


# === IFFT ===
class IFFT(FFT):
    """
    1-D inverse fast Fourier transform

    """

    def __init__(self):
        super.__init__()
        self.fft_fn = ifft_fn


class IFFT2(FFT):
    """
    2-D inverse fast Fourier transform

    """

    def __init__(self):
        super.__init__()
        self.fft_fn = ifft2_fn


class IRFFT(FFT):
    """
    1-D inverse fast Fourier transform for real-valued inputs

    """

    def __init__(self):
        super.__init__()
        self.fft_fn = irfft_fn

    def compute_output_spec(self, x):
        """
        Compute output spec of Fourier transform

        Parameters
        ----------
        x : KerasTensor | tuple | list
            Real- or complex input to FFT. A complex input must be composed of a tuple or list of the real- and imaginary part `(x_real, x_imag)`.

        Returns
        -------
        y_real_spec : KerasTensor
            spec of real part of `IRFFT(x)`

        """

        if not isinstance(x, (tuple, list)) or len(x) != 2:
            real = x
            imag = ops.zeros_like(x)
        else:
            real, imag = x
        # Both real and imaginary parts should have the same shape.
        if real.shape != imag.shape:
            raise ValueError(
                "Input `x` should be a tuple of two tensors - real and "
                "imaginary. Both the real and imaginary parts should have the "
                f"same shape. Received: x[0].shape = {real.shape}, "
                f"x[1].shape = {imag.shape}"
            )

        return KerasTensor(shape=real.shape, dtype=real.dtype)


class IRFFT2(IRFFT):
    """
    2-D inverse fast Fourier transform for real-valued inputs

    """

    def __init__(self):
        super.__init__()
        self.fft_fn = irfft2_fn


# === now make function wrapper ===
# === forward FFT ===
def fft(x):
    """
    1-D fast Fourier transform

    Parameters
    ----------
    x : KerasTensor | tuple | list
        Real- or complex input to FFT. A complex input must be composed of a tuple or list of the real- and imaginary part `(x_real, x_imag)`.

    Returns
    -------
    y_real, y_imag : (KerasTensor, KerasTensor)
        Tuple of real- and imaginary part of FFT(x).

    Examples
    --------
    >>> from ssp.keras.ops import fft
    >>> from keras import ops
    >>> signal = ops.array([-2, 8, 6, 4, 1, 0, 3, 5], dtype=float)
    >>> y_real, y_imag = fft(signal)
    >>> ops.convert_to_numpy(y_real)
    array([ 25.       ,   3.3639607, -10.       ,  -9.36396  ,  -9.       ,
        -9.36396  , -10.       ,   3.3639607], dtype=float32)
    >>> ops.convert_to_numpy(y_imag)
    array([ 0.       , -7.9497476,  1.       , -1.9497476,  0.       ,
        1.9497476, -1.       ,  7.9497476], dtype=float32)

    """

    if any_symbolic_tensors(cast_to_complex(x)):
        return FFT().symbolic_call(x)
    return fft_fn(x)


def fft2(x):
    """
    2-D fast Fourier transform

    Parameters
    ----------
    x : KerasTensor | tuple | list
        Real- or complex input to FFT. A complex input must be composed of a tuple or list of the real- and imaginary part `(x_real, x_imag)`.

    Returns
    -------
    y_real, y_imag : (KerasTensor, KerasTensor)
        Tuple of real- and imaginary part of FFT2(x).

    Examples
    --------
    >>> from ssp.keras.ops import fft2
    >>> from keras import ops
    >>> signal = ops.reshape(ops.array([-2, 8, 6, 12], dtype=float), (2, 2))
    >>> y_real, y_imag = fft2(signal)
    >>> ops.convert_to_numpy(y_real)
    array([
        [ 16.,  -8.],
        [ -4., -12.]
    ], dtype=float32)
    >>> ops.convert_to_numpy(y_imag)
    array([
        [0., 0.],
        [0., 0.]
    ], dtype=float32)

    """
    
    if any_symbolic_tensors(cast_to_complex(x)):
        return FFT2().symbolic_call(x)
    return fft2_fn(x)


def rfft(x):
    """
    1-D fast Fourier transform for real-valued inputs

    Parameters
    ----------
    x : KerasTensor
        Real-valued input to FFT.

    Returns
    -------
    y_real, y_imag : (KerasTensor, KerasTensor)
        Tuple of real- and imaginary part of RFFT(x).
        Both have size (n // 1 + 1) given ops.shape(x) == n

    Examples
    --------
    >>> from ssp.keras.ops import rfft
    >>> from keras import ops
    >>> signal = ops.array([-2, 8, 6, 4, 1, 0, 3, 5], dtype=float)
    >>> y_real, y_imag = rfft(signal)
    >>> ops.convert_to_numpy(y_real)
    array([ 25.       ,   3.3639607, -10.       ,  -9.36396  ,  -9.       ], dtype=float32)
    >>> ops.convert_to_numpy(y_imag)
    array([ 0.       , -7.9497476,  1.       , -1.9497476,  0.       ], dtype=float32)
       
    """

    if any_symbolic_tensors(cast_to_complex(x)):
        return RFFT().symbolic_call(x)
    return rfft_fn(x)


def rfft2(x):
    """
    2-D fast Fourier transform for real-valued inputs

    Parameters
    ----------
    x : KerasTensor
        Real-valued input to FFT.

    Returns
    -------
    y_real, y_imag : (KerasTensor, KerasTensor)
        Tuple of real- and imaginary part of RFFT2(x).
        Both have size (n, n // 1 + 1) given ops.shape(x) == (n, n)

    Examples
    --------
    >>> from ssp.keras.ops import rfft2
    >>> from keras import ops
    >>> signal = ops.reshape(ops.array([-2, 8, 6, 12, 4, 0, 2, 2, -2, 8, 6, 12, 4, 0, 2, 2], dtype=float), (4, 4))
    >>> y_real, y_imag = rfft2(signal)
    >>> ops.convert_to_numpy(y_real)
    array([
        [ 64., -12., -24.],
        [  0.,   0.,   0.],
        [ 32., -20., -40.],
        [  0.,   0.,   0.]
    ], dtype=float32)
    >>> ops.convert_to_numpy(y_imag)
    array([
        [ 0., 12.,  0.],
        [ 0.,  0.,  0.],
        [ 0.,  4.,  0.],
        [ 0.,  0.,  0.]
    ], dtype=float32)

    """

    if any_symbolic_tensors(cast_to_complex(x)):
        return RFFT2().symbolic_call(x)
    return rfft2_fn(x)


# === inverse FFT ===
def ifft(x):
    """
    1-D inverse fast Fourier transform

    Parameters
    ----------
    x : KerasTensor | tuple | list
        Real- or complex input to FFT. A complex input must be composed of a tuple or list of the real- and imaginary part `(x_real, x_imag)`.

    Returns
    -------
    y_real, y_imag : (KerasTensor, KerasTensor)
        Tuple of real- and imaginary part of IFFT(x).

    """

    if any_symbolic_tensors(cast_to_complex(x)):
        return IFFT().symbolic_call(x)
    return ifft_fn(x)


def ifft2(x):
    """
    2-D inverse fast Fourier transform

    Parameters
    ----------
    x : KerasTensor | tuple | list
        Real- or complex input to FFT. A complex input must be composed of a tuple or list of the real- and imaginary part `(x_real, x_imag)`.

    Returns
    -------
    y_real, y_imag : (KerasTensor, KerasTensor)
        Tuple of real- and imaginary part of IFFT2(x).

    """
    
    if any_symbolic_tensors(cast_to_complex(x)):
        return IFFT2().symbolic_call(x)
    return ifft2_fn(x)


def irfft(x, n: tuple = None):
    """
    1-D inverse fast Fourier transform for real-valued inputs

    Parameters
    ----------
    x : KerasTensor | tuple | list
        Real- or complex input to FFT. A complex input must be composed of a tuple or list of the real- and imaginary part `(x_real, x_imag)`.

    Returns
    -------
    y_real, y_imag : (KerasTensor, KerasTensor)
        Tuple of real- and imaginary part of IRFFT(x).
        Both have size (n // 1 + 1) given ops.shape(x) == n

    """
    
    if any_symbolic_tensors(cast_to_complex(x)):
        return IRFFT().symbolic_call(x, n=n)
    return irfft_fn(x, n=n)


def irfft2(x, n: tuple = None):
    """
    2-D inverse fast Fourier transform for real-valued inputs

    Parameters
    ----------
    x : KerasTensor | tuple | list
        Real- or complex input to FFT. A complex input must be composed of a tuple or list of the real- and imaginary part `(x_real, x_imag)`.

    Returns
    -------
    y_real, y_imag : (KerasTensor, KerasTensor)
        Tuple of real- and imaginary part of IRFFT(x).
        Both have size (n // 1 + 1, n) given ops.shape(x) == (n, n)

    """

    if any_symbolic_tensors(cast_to_complex(x)):
        return IRFFT2().symbolic_call(x, n=n)
    return irfft2_fn(x, n=n)
