from keras.src import ops
from keras.src.losses.loss import Loss
from keras.src.saving import serialization_lib
from keras.src.backend.config import backend
from .ops.helper import squeeze_or_expand_to_same_rank
from .ops import hard_lowpass, circular_hard_lowpass
from .ops.fft import fft, fft2, ifft, ifft2
from keras import KerasTensor
from functools import partial
from typing import Tuple


class FrequencyLossFunctionWrapper1D(Loss):
    def __init__(
        self,
        fn: callable,
        lowpass: str = None,
        f: KerasTensor = None,
        f_filter: float = 6.0,
        f_min: float = 0.0,
        p: float = 7.0,
        decay_start: int = 0,
        decay_epochs: int = 50,
        data_format: str = "channels_last",
        reduction: str = "sum_over_batch_size",
        name: str = None,
        **kwargs
    ):
        super().__init__(name=name, reduction=reduction, dtype="float32")

        self.data_format = data_format.lower()
        if self.data_format not in ["channels_first", "channels_last"]:
            raise ValueError(f"Unsupported data format {self.data_format}. Please choose from 'channels_first' and 'channel_last'.")
        
        self.lowpass = lowpass
        if (self.lowpass not in ['static', 'adaptive']) & (self.lowpass is not None):
            raise ValueError(f"Unsupported filter type, choose from {['static', 'adaptive']}")
        
        """
        For the frequency range calculations (mainly estimation of filter size) only the positive frequency range is required!
        """
        # self.use_lowpass: bool = True
        if f is not None:
            f = ops.convert_to_tensor(f)
        else:
            f = []

        self.f = ops.cast(f, dtype=self.dtype)
        self.nx = ops.shape(self.f)[0]

        self.f_filter = ops.cast(f_filter, dtype=self.dtype)
        self.f_min = ops.cast(f_min, dtype=self.dtype)
        self.p = ops.cast(p, dtype=self.dtype)
        
        # decay for loss filter (spectral radius is gradually reduced from self.nx // 2 + 1 (full frequency range) to the spectral radius that corresponds to the calculated lowpass frequency)
        self.decay_epochs: int = decay_epochs
        self.decay_start: int = decay_start
        self.epoch: int = None  # has to be initialized via callback (UseLossLowpassDecay callback)

        # define callables
        self.fn = fn
        self._fn_kwargs = kwargs

        self.fft: callable = fft
        self.ifft: callable = ifft
        self.norm: callable = partial(ops.linalg.norm, axis=-1)
        self.expand_dims: callable = ops.expand_dims

    def call(self, y_true: KerasTensor, y_pred: KerasTensor) -> KerasTensor:
        y_true = ops.convert_to_tensor(y_true)  # shape = (batch, x, ch) if data_format == 'channels_last' else (batch, x) or (batch, 1, x)
        y_pred = ops.convert_to_tensor(y_pred)

        # squeeze along channel axis
        y_true, y_pred = squeeze_or_expand_to_same_rank(y_true, y_pred, axis=-1 if self.data_format == 'channels_last' else 1)  # this adds channel dimension to y_true if there is none

        # make channels first for FFT application
        y_true = self.transpose_to_channels_first(y_true)  # shape = (batch, ch, x) or (batch, x)
        y_pred = self.transpose_to_channels_first(y_pred)

        if self.lowpass:  # i.e. lowpass is not None
            """ apply frequency filter to ground truth data
            note: we are working with tuples, since keras has no complex dtype
            """
            y_true_real, y_true_imag = self.fft(y_true)
            y_true_real, y_true_imag = self.apply_filter(real=y_true_real, imag=y_true_imag)

            y_true, _ = self.ifft((y_true_real, y_true_imag))

        return self.fn(y_true, y_pred, **self._fn_kwargs)
    
    # === everything that has to do with filtering / Fourier domain ===
    def estimate_peak_frequency(self, power_spectrum: KerasTensor) -> KerasTensor:
        """
        ## Estimate peak frequency from spectrum
        ---
        ### Args:
            power_spectrum (KerasTensor): power spectrum of ground truth

        ### Returns:
            KerasTensor: a tensor with shape==(batch_size, ) with peak frequency (capped at self.f_min) for each sample
        """
        peak_frequency = ops.divide_no_nan(
            self.norm(ops.abs(ops.multiply(self.f, power_spectrum))),
            self.norm(power_spectrum)
        )
        return ops.where(
            (peak_frequency < self.f_min) | ops.numpy.isnan(peak_frequency),
            self.f_min, 
            peak_frequency
        )
    
    def get_f_hat(self, real: KerasTensor, imag: KerasTensor) -> KerasTensor:
        """
        ## Normalize frequency range self.f by estimated peak frequency
        ---
        ### Args:
            real (KerasTensor): real part of Fourier transform of signal y
            imag (KerasTensor): imaginary part of Fourier transform of signal y

        ### Returns:
            k.KerasTensor: frequency range normalized by estimated peak frequency
        """
        ps = self.magnitude(real=real, imag=imag)
        normalized_ps = (ps - ops.min(ps)) / (ops.max(ps) - ops.min(ps))
        
        ps = ops.cast(normalized_ps ** self.p, dtype=real.dtype)
        peak_frequency = self.estimate_peak_frequency(power_spectrum=ps)
        return ops.divide_no_nan(
            ops.expand_dims(self.f, axis=0),  # add batch dimension
            self.expand_dims(peak_frequency, axis=-1)  # add space dimension
        )
    
    def get_frequency_filter(self, real: KerasTensor, imag: KerasTensor) -> KerasTensor:
        """
        The frequency filter is implemented as a tukey window, which is multiplied with the Fourier spectrum.
        The tukey window requires
            1. the overall length (int), and 
            2. the length of the window, which is here given by the index where the filter frequency exceeds the frequency vector

        The radius of the filter is calculated based on the frequency grid 'f' and 'f_filter'.
        The closest frequency component to 'f_filter' is found by 'diff = abs(f - f_filter)'.
        The index of the minimum entry in 'diff' is used to calculate the radius of the filter.
        
        All calculations are performed on the positive quadrant of f.
        For the adaptive filter, the frequency spectrum scaled by the peak frequency is used.
        """
        nx = self.nx // 2
        if self.lowpass == 'adaptive':
            # work with the freqency range normalized by the peak frequency
            # NOTE this is of shape (b, N)!
            f = self.get_f_hat(real=real, imag=imag)

            # reduce to positive frequencies
            f = f[tuple([slice(None), slice(None, nx)])]

            # use only positive half of f
            spectral_radius = ops.argmin(ops.abs(f - self.f_filter), axis=-1)
            freq_filter = ops.vectorized_map(self.lowpass_fn, spectral_radius)

            # apply fft shift to filter
            return self.fftshift(freq_filter, axis=-1)
        else:
            # work with the defaults frequency range
            # NOTE this is of shape (N,)

            # reduce to positive frequencies
            f = self.f[slice(None, nx)]

            spectral_radius = ops.argmin(ops.abs(f - self.f_filter))
            freq_filter = self.lowpass_fn(spectral_radius=spectral_radius)

            # apply fft shift
            return self.fftshift(freq_filter)

    def lowpass_decay(self, spectral_radius: int) -> int:
        """
        We want to have a linear decrease in spectral radius over the epochs,
        starting from full frequency range to spectral radius.
        The linear function is consequently given by
            s = int(spectral_radius + alpha*(s - spectral_radius))
            with
                s: spectral radius(epoch)
                alpha: slope of linear function, i.e., (epoch - self.decay_start) / self.decay_epochs
        """
        if self.epoch is None:
            # callback to steer decay is not active, just return spectral radius
            return spectral_radius
        
        if self.epoch > (self.decay_start + self.decay_epochs):
            return spectral_radius
        
        max_radius = self.nx // 2 - 1  # zero indexing, we need 255 --> positive frequencies only

        if self.epoch < self.decay_start:
            return max_radius
        
        alpha = ops.divide(self.epoch - self.decay_start, self.decay_epochs)
        return ops.cast(max_radius + alpha * ops.cast(spectral_radius - max_radius, dtype=alpha.dtype), dtype=spectral_radius.dtype)
    
    def apply_filter(self, real: KerasTensor, imag: KerasTensor) -> Tuple[KerasTensor, KerasTensor]:
        """
        ## Apply frequency filter
        ---
        ### Args:
            real (KerasTensor): real part of Fourier transform of signal y
            imag (KerasTensor): imaginary part of Fourier transform of signal y

        ### Returns:
            KerasTensor: Fourier transform of signal y with frequencies > f_filter set to (0 + 0j)
        """
        frequency_filter = self.get_frequency_filter(real, imag)
        return ops.multiply(real, frequency_filter), ops.multiply(imag, frequency_filter)
    
    def transpose_to_channels_first(self, inputs: KerasTensor) -> KerasTensor:
        if self.data_format == 'channels_first':
            return inputs
        
        shape = ops.shape(inputs)
        if len(shape) == 2:  # there is no channel dimension!
            return inputs
        
        transpose_axes = list(range(len(shape)))
        # move channel_dimension to first position after batch size ('channels_first')
        ch_dim = transpose_axes.pop(-1)
        transpose_axes.insert(-1, ch_dim)  # NOTE this is only for 1d data; -1 inserts at index -2!

        return ops.transpose(inputs, axes=transpose_axes)
    
    def lowpass_fn(self, spectral_radius: int):
        # adjust spectral radius
        spectral_radius = self.lowpass_decay(spectral_radius=spectral_radius)
        return hard_lowpass(n=self.nx, spectral_radius=spectral_radius)
    
    def fftshift(self, x: KerasTensor, axis: int = None) -> KerasTensor:
        return ops.roll(x, shift=self.nx // 2, axis=axis)
    
    @staticmethod
    def magnitude(real: KerasTensor, imag: KerasTensor):
        return ops.sqrt(ops.square(real) + ops.square(imag))
    
    # === serialization ===
    def get_config(self) -> dict:
        base_config: dict = super().get_config()
        config = {
            # "fn": serialization_lib.serialize_keras_object(self.fn),
            "lowpass": self.lowpass,
            "f": self.tolist(self.f),
            "f_filter": float(self.f_filter),
            "f_min": float(self.f_min),
            "p": float(self.p),
            "decay_epochs": self.decay_epochs,
            "decay_start": self.decay_start,
            "data_format": str(self.data_format)
        }
        config.update(serialization_lib.serialize_keras_object(self._fn_kwargs))
        return {**base_config, **config}
    
    @classmethod
    def from_config(cls, config: dict):
        f = config.pop("f")
        f = ops.array(f)
        # if "fn" in config:
        #     config = serialization_lib.deserialize_keras_object(config)
        return cls(f=f, **config)
    
    # === helper routines ===
    def transpose_to_channels_first(self, inputs: KerasTensor) -> KerasTensor:
        if self.data_format == "channels_first":
            return inputs
        
        if not hasattr(self, "transpose_axes"):
            # get transpose axes once and cache it
            shape = ops.shape(inputs)
            if len(shape) == 2:  # there is no channel dimension!
                return inputs
            
            transpose_axes = list(range(len(shape)))
            # move channel_dimension to first position after batch size ('channels_first')
            ch_dim = transpose_axes.pop(-1)
            transpose_axes.insert(-1, ch_dim)  # NOTE this is only for 1d data; -1 inserts at index -2!

            self.transpose_axes = transpose_axes

        return ops.transpose(inputs, axes=self.transpose_axes)
    
    @staticmethod
    def tolist(arr: KerasTensor) -> list:
        if backend() == "tensorflow":
            shape = arr.get_shape()
            if len(shape) == 2:
                return list([item.numpy().tolist() for item in arr])
            return arr.numpy().tolist()
        
        if backend() == "jax":
            if arr.ndim == 2:
                return list([item.tolist() for item in arr])
            return arr.tolist()
        
        raise RuntimeError(f"{backend()} is not supported! SSP currently only works with 'jax' and 'tensorflow' backends.")


class FrequencyLossFunctionWrapper2D(FrequencyLossFunctionWrapper1D):
    def __init__(
        self,
        fn: callable,
        lowpass: str = None,
        f: KerasTensor = None,
        f_filter: float = 6.0,
        f_min: float = 0.0,
        p: float = 7.0,
        data_format: str = "channels_last",
        reduction: str = "sum_over_batch_size",
        name: str = None,
        **kwargs
    ):
        super().__init__(
            fn=fn,
            lowpass=lowpass,
            f=f,
            f_filter=f_filter,
            f_min=f_min,
            p=p,
            data_format=data_format,
            reduction=reduction,
            name=name,
            **kwargs
        )

        # overwrite callables for 2D
        self.fft: callable = fft2
        self.ifft: callable = ifft2
        self.norm: callable = partial(ops.linalg.norm, axis=(-2, -1))
        self.expand_dims: callable = lambda x, axis: ops.expand_dims(ops.expand_dims(x, axis=axis), axis=axis)

        """
        For the frequency range calculations (mainly estimation of filter size) only the positive frequency range is required!
        Here, f is a (potentially empty) 1D tensor!
        """
        if ops.convert_to_numpy(self.f).ndim == 1:
            self.f: KerasTensor = self.magnitude(*ops.meshgrid(self.f, self.f))
        self.ny, self.nx = ops.shape(self.f)

    # === everything that has to do with filtering / Fourier domain adjusted for 2D ===
    def get_frequency_filter(self, real: KerasTensor, imag: KerasTensor) -> KerasTensor:
        """
        The frequency filter is implemented as a tukey window, which is multiplied with the Fourier spectrum.
        The tukey window requires
            1. the overall length (int), and 
            2. the length of the window, which is here given by the index where the filter frequency exceeds the frequency vector

        The radius of the filter is calculated based on the frequency grid 'f' and 'f_filter'.
        The closest frequency component to 'f_filter' is found by 'diff = abs(f - f_filter)'.
        The index of the minimum entry in 'diff' is used to calculate the radius of the filter.
        
        All calculations are performed on the positive quadrant of f.
        For the adaptive filter, the frequency spectrum scaled by the peak frequency is used.
        """
        nx = self.nx // 2
        ny = self.ny // 2
        if self.lowpass == 'adaptive':
            # work with the freqency range normalized by the peak frequency
            # NOTE this is of shape (b, N, N)!
            f = self.get_f_hat(real=real, imag=imag)

            # reduce to positive frequencies
            f = f[tuple([slice(None), slice(None, ny), slice(None, nx)])]

            diff = abs(f - self.f_filter)

            # flatten tensor
            diff = ops.reshape(diff, newshape=(-1, ny * nx))
            min_idx = ops.argmin(diff, axis=-1)

            rows = min_idx // ny
            cols = min_idx % nx

            spectral_radius = ops.cast(ops.round(self.magnitude(rows, cols)), dtype='int32')
            freq_filter = ops.vectorized_map(self.lowpass_fn, spectral_radius)

            # apply fft shift to filter
            return self.fftshift(freq_filter)
        else:
            # work with the defaults frequency range
            # NOTE this is of shape (N,N)

            # reduce to positive frequencies
            f = self.f[tuple([slice(None, ny), slice(None, nx)])]

            min_idx = ops.argmin(ops.abs(f - self.f_filter))  # we are interested in the minimum in x-direction
            
            rows = min_idx // ny
            cols = min_idx % nx

            spectral_radius = ops.cast(ops.round(self.magnitude(rows, cols)), dtype='int32')
            freq_filter = self.lowpass_fn(spectral_radius=spectral_radius)

            # apply fft shift
            return self.fftshift(freq_filter)
        
    def get_config(self) -> dict:
        """
        while 2D loss does not support LowpassDecay, this function has to be implemented without arguments
            decay_epochs
            decay_start
        """
        config: dict = super().get_config()
        config.pop("decay_epochs", None)
        config.pop("decay_start", None)

        return config
    
    # === helper routines adjusted for 2D ===
    def transpose_to_channels_first(self, inputs: KerasTensor) -> KerasTensor:
        if self.data_format == 'channels_first':
            return inputs
        
        shape = ops.shape(inputs)
        if len(shape) == 3:  # there is no channel dimension!
            return inputs
        
        transpose_axes = list(range(len(shape)))
        # move channel_dimension to first position after batch size ('channels_first')
        ch_dim = transpose_axes.pop(-1)
        transpose_axes.insert(-2, ch_dim)  # NOTE this is only for 2d data

        return ops.transpose(inputs, axes=transpose_axes)
    
    def lowpass_decay(self, spectral_radius: int) -> int:
        """
        We want to have a linear decrease in spectral radius over the epochs,
        starting from full frequency range to spectral radius.
        The linear function is consequently given by
            s = int(spectral_radius + alpha*(s - spectral_radius))
            with
                s: spectral radius(epoch)
                alpha: slope of linear function, i.e., (epoch - self.decay_start) / self.decay_epochs
        """
        if self.epoch is None:
            # callback to steer decay is not active, just return spectral radius
            return spectral_radius
        
        if self.epoch > (self.decay_start + self.decay_epochs):
            return spectral_radius
        
        max_radius = ops.cast(ops.round(self.magnitude(self.nx // 2 - 1, self.ny // 2 - 1)), dtype='int32')  # self.nx // 2 - 1  # zero indexing, we need 255 --> positive frequencies only

        if self.epoch < self.decay_start:
            return max_radius
        
        alpha = ops.divide(self.epoch - self.decay_start, self.decay_epochs)
        return ops.cast(max_radius + alpha * ops.cast(spectral_radius - max_radius, dtype=alpha.dtype), dtype=spectral_radius.dtype)
    
    def lowpass_fn(self, spectral_radius: int):
        spectral_radius = self.lowpass_decay(spectral_radius=spectral_radius)
        return circular_hard_lowpass(n=self.nx, spectral_radius=spectral_radius)
    
    def fftshift(self, x: KerasTensor) -> KerasTensor:
        return ops.roll(ops.roll(x, shift=self.ny // 2, axis=-2), shift=self.nx // 2, axis=-1)
