import tensorflow as tf
from typing import Tuple
from functools import partial


def _get_complex_tensor_from_tuple(x) -> tf.Tensor:
    if not isinstance(x, (tuple, list)) or len(x) != 2:
        real = x
        imag = tf.zeros_like(x)
    else:
        real, imag = x

    complex_input = tf.complex(real=real, imag=imag)
    return complex_input


def _get_real_tensor_from_tuple(x) -> tf.Tensor:
    if not isinstance(x, (tuple, list)) or len(x) != 2:
        return x
    
    real, _ = x
    return real


# === FFT ===
def _fft(x: tuple, fn: callable) -> Tuple[tf.Tensor, tf.Tensor]:
    complex_input = _get_complex_tensor_from_tuple(x)
    complex_output = fn(complex_input)
    return tf.math.real(complex_output), tf.math.imag(complex_output)


# === real valued FFT ===
def _rfft(x: tf.Tensor, fn: callable) -> Tuple[tf.Tensor, tf.Tensor]:
    complex_output = fn(x)
    return tf.math.real(complex_output), tf.math.imag(complex_output)


def _irfft(x: tf.Tensor, fn: callable, n: tuple = None) -> tf.Tensor:
    complex_input = _get_complex_tensor_from_tuple(x)
    complex_output = fn(complex_input, fft_length=n)
    return tf.math.real(complex_output), tf.math.imag(complex_output)

# === derived functions
def fft_fn(x):
    return partial(_fft, fn=tf.signal.fft)(x)


def fft2_fn(x):
    return partial(_fft, fn=tf.signal.fft2d)(x)


def ifft_fn(x):
    return partial(_fft, fn=tf.signal.ifft)(x)


def ifft2_fn(x):
    return partial(_fft, fn=tf.signal.ifft2d)(x)


def rfft_fn(x):
    return partial(_rfft, fn=tf.signal.rfft)(x)


def rfft2_fn(x):
    return partial(_rfft, fn=tf.signal.rfft2d)(x)


def irfft_fn(x, n: tuple = None):
    y_real, _ = partial(_irfft, fn=tf.signal.irfft, n=n)(x)
    return y_real


def irfft2_fn(x, n: tuple = None):
    y_real, _ = partial(_irfft, fn=tf.signal.irfft2d, n=n)(x)
    return y_real
