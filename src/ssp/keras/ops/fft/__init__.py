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


# === forward FFT ===
class FFT(Operation):
    def __init__(
            self,
    ):
        super().__init__()
        self.fft_fn = fft_fn

    def compute_output_spec(self, x):
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
        return self.fft_fn(x)
    

class FFT2(FFT):
    def __init__(
            self
    ):
        super.__init__()
        self.fft_fn = fft2_fn


class RFFT(FFT):
    def __init__(
            self
    ):
        super.__init__()
        self.fft_fn = rfft_fn


class RFFT2(FFT):
    def __init__(
            self
    ):
        super.__init__()
        self.fft_fn = rfft2_fn


# === inverse FFT ===
class IFFT(FFT):
    def __init__(
            self
    ):
        super.__init__()
        self.fft_fn = ifft_fn


class IFFT2(FFT):
    def __init__(
            self
    ):
        super.__init__()
        self.fft_fn = ifft2_fn


class IRFFT(FFT):
    def __init__(
            self
    ):
        super.__init__()
        self.fft_fn = irfft_fn

    def compute_output_spec(self, x):
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
    def __init__(
            self
    ):
        super.__init__()
        self.fft_fn = irfft2_fn


# === now make function wrapper ===
# === forward FFT ===
def fft(x):
    if any_symbolic_tensors(cast_to_complex(x)):
        return FFT().symbolic_call(x)
    return fft_fn(x)


def fft2(x):
    if any_symbolic_tensors(cast_to_complex(x)):
        return FFT2().symbolic_call(x)
    return fft2_fn(x)


def rfft(x):
    if any_symbolic_tensors(cast_to_complex(x)):
        return RFFT().symbolic_call(x)
    return rfft_fn(x)


def rfft2(x):
    if any_symbolic_tensors(cast_to_complex(x)):
        return RFFT2().symbolic_call(x)
    return rfft2_fn(x)


# === inverse FFT ===
def ifft(x):
    if any_symbolic_tensors(cast_to_complex(x)):
        return IFFT().symbolic_call(x)
    return ifft_fn(x)


def ifft2(x):
    if any_symbolic_tensors(cast_to_complex(x)):
        return IFFT2().symbolic_call(x)
    return ifft2_fn(x)


def irfft(x, n: tuple = None):
    if any_symbolic_tensors(cast_to_complex(x)):
        return IRFFT().symbolic_call(x, n=n)
    return irfft_fn(x, n=n)


def irfft2(x, n: tuple = None):
    if any_symbolic_tensors(cast_to_complex(x)):
        return IRFFT2().symbolic_call(x, n=n)
    return irfft2_fn(x, n=n)
