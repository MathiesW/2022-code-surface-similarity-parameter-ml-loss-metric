"""
%---------------------------------------------------------------------------------------------
% For Paper,
% "Surface Similarity Parameter: A new machine learning loss metric for oscillatory spatio-temporal data"
% by Mathies Wedler and Merten Stender and Marco Klein and Svenja Ehlers and Norbert Hoffmann
% Copyright (c) Dynamics Group, Hamburg University of Technology. All rights reserved.
% Licensed under the GPLv3. See LICENSE in the project root for license information.
% Method is based on the work
% "Marc Perlin and Miguel D. Bustamante,
% A robust quantitative comparison criterion of two signals based on the Sobolev norm of their difference,
% Journal of Engineering Mathematics, volume 101,
% DOI 10.1007/s10665-016-9849-7"
%--------------------------------------------------------------------------------------------
"""


from abc import ABC
import tensorflow as tf
import numpy as np
from typing import Union


class _SurfaceSimilarityParameterBase(tf.keras.losses.Loss, ABC):
    def __init__(
            self,
            dimension: int,
            name: str = 'SSP',
            reduction: tf.keras.losses.Reduction = tf.keras.losses.Reduction.AUTO,
            **kwargs):
        """
        Surface similarity parameter base class
        :param dimension: dimension of data, 1 or 2
        :param name: name of loss function, default: "SSP"
        :param reduction: reduction for tf.keras.losses.Loss base class (cf. https://www.tensorflow.org/api_docs/python/tf/keras/losses/Loss)
        :param kwargs: keyword arguments for tf.keras.losses.Loss base class (cf. https://www.tensorflow.org/api_docs/python/tf/keras/losses/Loss)
        """
        assert dimension in [1, 2]

        super().__init__(name=name, reduction=reduction, **kwargs)
        self.dimension = dimension
        self.my_fft = tf.signal.fft if dimension == 1 else tf.signal.fft2d

    @staticmethod
    def trapezoidal_tf(y: tf.Tensor) -> tf.Tensor:
        """
        trapezoidal integration along last axis defined in terms of TF tensors
        :param y: tensor, y.shape = (batch, x)
        :return: integral over y
        """
        tr1 = tf.divide(y[..., 0] + y[..., -1], 2.0)
        tr2 = tf.reduce_sum(y[..., 1:-1], axis=-1)
        return tf.reduce_sum((tr1, tr2), axis=0)

    def sobolev_norm(self, y_f: tf.Tensor) -> tf.Tensor:
        """
        Sobolev norm defined in terms of TF tensors
        :param y_f: spectrum of tensor y, y.shape = (batch, n_grid)
        :return: Sobolev norm of y_f
        """
        y_f = tf.square(tf.abs(y_f))

        for _ in range(self.dimension):
            y_f = self.trapezoidal_tf(y_f)

        return tf.sqrt(y_f)

    @tf.function
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        assert y_true.shape == y_pred.shape, f'Shape of tensors do not match: {y_true.shape} vs {y_pred.shape}'

        y_true_f = self.my_fft(tf.cast(y_true, dtype=tf.complex64))
        y_pred_f = self.my_fft(tf.cast(y_pred, dtype=tf.complex64))

        return tf.math.divide_no_nan(
            self.sobolev_norm(tf.subtract(y_true_f, y_pred_f)),
            tf.add(self.sobolev_norm(y_true_f), self.sobolev_norm(y_pred_f))
        )


class _SurfaceSimilarityParameterLowPassBase(_SurfaceSimilarityParameterBase):
    def __init__(
            self,
            dimension: int,
            k: Union[np.ndarray, tf.constant],
            k_filter: float = None,
            lowpass: str = 'adaptive',
            p: float = 8.0,
            **kwargs
    ):
        assert lowpass in ['static', 'adaptive']
        name = '-'.join(['SSP' if dimension == 1 else 'SSP2D', lowpass])
        super().__init__(dimension=dimension, name=name, **kwargs)

        self.k = tf.cast(k, dtype=tf.float32)
        self.p = tf.cast(p, dtype=tf.float32)

        if lowpass == 'static':
            self.k_filter = k_filter or 2.0  # Hz
            self.static_filter = self.get_static_filter()

        if lowpass == 'adaptive':
            self.k_filter = k_filter or 6.0
            self.static_filter = None

    def get_adaptive_filter(self, y_f: tf.Tensor) -> tf.Tensor:
        return NotImplemented

    def get_static_filter(self) -> tf.Tensor:
        return NotImplemented

    @tf.function
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        assert y_true.shape == y_pred.shape, f'Shape of tensors do not match: {y_true.shape} vs {y_pred.shape}'
        y_true = tf.cast(y_true, dtype=tf.complex64)
        y_pred = tf.cast(y_pred, dtype=tf.complex64)

        y_true_f = self.my_fft(tf.expand_dims(y_true, axis=0) if tf.rank(y_true) == self.dimension else y_true)
        y_pred_f = self.my_fft(tf.expand_dims(y_pred, axis=0) if tf.rank(y_pred) == self.dimension else y_pred)

        # apply filter to Fourier spectrum of ground truth data
        y_true_f *= self.get_adaptive_filter(y_true_f) if self.static_filter is None else self.static_filter

        return tf.math.divide_no_nan(
            self.sobolev_norm(tf.subtract(y_true_f, y_pred_f)),
            tf.add(
                self.sobolev_norm(y_true_f),
                self.sobolev_norm(y_pred_f)
            )
        )


class SurfaceSimilarityParameter(_SurfaceSimilarityParameterBase):
    def __init__(self, **kwargs):
        """
        Surface similarity parameter (SSP) for one-dimensional data.
        The SSP expects the data format NW with no channel dimension

        :param kwargs: keyword arguments for tf.keras.losses.Loss base class (cf. https://www.tensorflow.org/api_docs/python/tf/keras/losses/Loss)
        """
        super().__init__(dimension=1, **kwargs)


class SurfaceSimilarityParameterLowPass(_SurfaceSimilarityParameterLowPassBase):
    def __init__(
            self,
            k: Union[np.ndarray, tf.constant],
            k_filter: float = None,
            lowpass: str = 'adaptive',
            p: float = 8.0,
            **kwargs
    ):
        """
        Surface similarity parameter loss function for one-dimensional data with adaptive low-pass filter.
        The low pass filter is applied to the ground truth data only, such that the model is forced to suppress the
        HF range in order to minimize the loss.

        The peak wave number kp is estimated from the Fourier spectrum of the ground truth data
            kp = (integral(k * fft(y)**p)) / (integral(fft(y)**p)
        with the exponent p usually between 6.0 -- 8.0, cf. Sobey and Young, 1986 (doi:10.1061/(asce)0733-950x(1986)112:3(370))
        :param k: wave number or frequency vector
        :param k_filter: frequency at which filter becomes active, i.e. S(k > k_filter) := 0
        :param lowpass: choose from adaptive or static
        :param p: exponent to estimate kp from Fourier spectrum of ground truth
        :param name: name of loss function, default: "SSP adaptive"
        :param kwargs: keyword arguments for tf.keras.losses.Loss (cf. https://www.tensorflow.org/api_docs/python/tf/keras/losses/Loss)
        """
        super().__init__(dimension=1, k=k, k_filter=k_filter, lowpass=lowpass, p=p, **kwargs)

    def get_adaptive_filter(self, y_f: tf.Tensor) -> tf.Tensor:
        spec = tf.cast(tf.abs(y_f), dtype=tf.float32) ** self.p
        kp = tf.cast(
            tf.divide(
                self.trapezoidal_tf(tf.abs(self.k * spec)),
                self.trapezoidal_tf(spec)
            ), dtype=tf.float32
        )

        k_hat = tf.divide(tf.expand_dims(self.k, axis=0), tf.expand_dims(kp, axis=1))

        return tf.cast(
            tf.where(
                tf.greater_equal(tf.abs(k_hat), tf.cast(tf.ones_like(k_hat) * self.k_filter, dtype=tf.float32)),
                0,
                1
            ), dtype=tf.complex64
        )

    def get_static_filter(self) -> tf.Tensor:
        return tf.cast(tf.where(tf.greater_equal(tf.abs(self.k), self.k_filter), 0, 1), dtype=tf.complex64)


class SurfaceSimilarityParameter2D(_SurfaceSimilarityParameterBase):
    def __init__(self, **kwargs):
        """
        Surface similarity parameter (SSP) for two-dimensional data.
        The SSP expects the data format NW with no channel dimension

        :param kwargs: keyword arguments for tf.keras.losses.Loss base class (cf. https://www.tensorflow.org/api_docs/python/tf/keras/losses/Loss)
        """
        super().__init__(dimension=2, **kwargs)


class SurfaceSimilarityParameter2DLowPass(_SurfaceSimilarityParameterLowPassBase):
    def __init__(
            self,
            k: Union[np.ndarray, tf.constant],
            k_filter: float = None,
            lowpass: str = 'adaptive',
            p: float = 8.0,
            **kwargs
    ):
        super().__init__(dimension=2, k=k, k_filter=k_filter, lowpass=lowpass, p=p, **kwargs)
        self.grid = tf.cast(tf.sqrt(sum(tf.square(tf.meshgrid(self.k, self.k)))), dtype=tf.float32)

    def get_adaptive_filter(self, y_f: tf.Tensor) -> tf.Tensor:
        # for square domain only! augmentation for 2D of eq. 2.14 (Klein)
        kp = tf.cast(
            tf.divide(
                self.trapezoidal_tf(self.trapezoidal_tf(tf.abs(self.k * tf.abs(y_f) ** self.p))),
                self.trapezoidal_tf(self.trapezoidal_tf(tf.abs(y_f) ** self.p))
            ),
            dtype=tf.float32
        )
        k_hat = tf.divide(tf.expand_dims(self.grid, axis=0), tf.expand_dims(tf.expand_dims(kp, axis=-1), axis=-1))

        return tf.cast(
            tf.where(
                tf.greater_equal(k_hat, self.k_filter),
                0,
                1
            ), dtype=tf.complex64
        )

    def get_static_filter(self) -> tf.Tensor:
        return tf.cast(
                tf.where(
                    tf.greater_equal(self.grid, self.k_filter),
                    0,
                    1
                ), dtype=tf.complex64
            )
