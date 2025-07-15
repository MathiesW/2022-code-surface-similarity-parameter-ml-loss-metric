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
    """
    Base class FrequencyLossFunctionWrapper1D to implement new loss functions with the option to apply a frequency filter to the ground truth.
    This frequency filter helps the model to focus on the relevant frequency range without the need to, e.g., remove HF noise in additional preprocessing steps.

    There are two lowpass filters to choose from, the "static" and the "adaptive" lowpass.
    The "static" lowpass defines a global cut-off frequency at 'f_filter'.
    The "adaptive" lowpass analyzes the ground truth data, extracts the peak frequency, and sets a dynamic cut-off frequency at for each sample.
    The parameter 'f_filter' becomes a multiplier for the peak frequency, after which the frequency components are suppressed.

    The definition of a lowpass {"static", "adaptive"} requires a frequency range 'f'.
    It enables an additional step in the loss calculation, where 
        1. the ground truth is transformed via 1-D FFT,
        2. a hard binary lowpass filter is applied to the Fourier spectrum to set all frequencies f>f_filter to 0+0j,
        3. the filtered ground truth is transformed back to its initial space.
    If lowpass==None, the FFT calculation is skipped, and no 'f' is required.

    This class inherits from keras.losses.Loss and can thus be used directly in model.compile()

    Parameters
    ----------
    fn : callable
        Definition of the loss function.
        The function has to accept two tensors (y_true and y_pred) and return a float.
    lowpass : str, optional {None, "static", "adaptive"}
        Lowpass filter that is applied to the ground truth in order to suppress the higher frequency range 'f>f_filter'.
        Defaults to None.
    f : array_like, optional
        Frequency range for the data.
        Is required once a lowpass {"static", "adaptive"} is used.
        Defaults to None.
    f_filter : float, optional
        Threshold for the lowpass filter.
        With the static lowpass, the ground truth spectrum is set to 0+j0 for 'f>f_filter'.
        With the adaptive lowpass, the ground truth spectrum is set to 0+j0 for 'f>f_p * f_filter',
        where f_p is the peak frequency that is automatically derived from the ground truth spectrum.
        Defaults to 6.0.
    f_min : float, optional
        Cap for the lowest peak frequency for cases when the automatic estimation of the peak frequency fails (estimated 'f_p < 0' or 'f_p == NaN').
        Defaults to 0.0.
    p : float, optional
        Exponent to weigh the spectrum towards the peak frequency (for the estimation of the peak frequency), c.f. 
            Mansard & Funke, "On the fitting of parametric models to measured wave spectra" (1988), and
            Sobey & Young, "Hurricane Wind Waves---A discrete spectral model" (1986), https://ascelibrary.org/doi/10.1061/%28ASCE%290733-950X%281986%29112%3A3%28370%29
        Defaults to 7.0.
    decay_start : int, optional !!!Requires 'UseLossLowpassDecay' callback to work!!!
        Epoch from which on the lowpass filter is linearly decreased from 0 to 'f_filter'.
        Defaults to 0.
    decay_epochs : int, optional !!!Requires 'UseLossLowpassDecay' callback to work!!!
        Number of epochs over which the lowpass filter is linearly decreased from 0 to 'f_filter'.
        Defaults to 50.
    data_format : str, optional {"channels_last", "channels_first"}
        The ordering of the dimensions in the inputs:
        "channels_last" corresponds to inputs with shape (batch_size, *dims, channels), 
        "channels_first" corresponds to inputs with shape (batch_size, channels, *dims).
        Defaults to "channels_last".
    reduction : str, optional {"sum_over_batch_size", "None", "auto", "sum"}
        Type of reduction to apply to the loss.
        In almost all cases this should be `"sum_over_batch_size"`.
        Supported options are `"sum"`, `"sum_over_batch_size"` or `None`.
    name : str, optional
        Name of the loss function. Name is inhereted from class name if name=None.
        Defaults to None.
    **kwargs
        Additional keyword arguments for fn.

    Notes
    -----
    Both the "adaptive" and "static" lowpass filter can be linearly increased from 0 to f_filter over 'decay_epoch' epochs, starting at epoch 'decay_start'.
    For this to work, the training has to be conducted using the 'UseLossLowpassDecay' callback, which sets the class variable 'self.epoch' to the current training epoch.
    See examples of SSP1D and SSP2D for a MWE.

    """

    def __init__(
        self,
        fn,
        lowpass=None,
        f=None,
        f_filter=6.0,
        f_min=0.0,
        p=7.0,
        decay_start=0,
        decay_epochs=50,
        data_format="channels_last",
        reduction="sum_over_batch_size",
        name=None,
        **kwargs
    ):
        super().__init__(name=name, reduction=reduction, dtype="float32")

        self.data_format = data_format.lower()
        if self.data_format not in ["channels_first", "channels_last"]:
            raise ValueError(f"Unsupported data format {self.data_format}. Please choose from 'channels_first' and 'channel_last'.")
        
        self.lowpass = lowpass
        if (self.lowpass not in ['static', 'adaptive']) & (self.lowpass is not None):
            raise ValueError(f"Unsupported filter type, choose from {['static', 'adaptive']}")
        
        # For the frequency range calculations (mainly estimation of filter size) only the positive frequency range is required! 
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
        self.decay_epochs = decay_epochs
        self.decay_start = decay_start
        self.epoch = None  # has to be initialized via callback (UseLossLowpassDecay callback)

        # define callables
        self.fn = fn
        self._fn_kwargs = kwargs

        self.fft: callable = fft
        self.ifft: callable = ifft
        self.norm: callable = partial(ops.linalg.norm, axis=-1)
        self.expand_dims: callable = ops.expand_dims

    def call(self, y_true, y_pred):
        """
        Call method of FrequencyLossFunctionWrapper1D

        Parameters
        ----------
        y_true : KerasTensor
            Ground truth
        y_pred : KerasTensor
            Prediction

        Returns
        -------
        loss : KerasTensor
            The scalar loss calculated from 'y_true' and 'y_pred' using 'self.fn'.

        """

        y_true = ops.convert_to_tensor(y_true)  # shape = (batch, x, ch) if data_format == 'channels_last' else (batch, x) or (batch, 1, x)
        y_pred = ops.convert_to_tensor(y_pred)

        # squeeze along channel axis
        y_true, y_pred = squeeze_or_expand_to_same_rank(y_true, y_pred, axis=-1 if self.data_format == 'channels_last' else 1)  # this adds channel dimension to y_true if there is none

        # make channels first for FFT application
        y_true = self.transpose_to_channels_first(y_true)  # shape = (batch, ch, x) or (batch, x)
        y_pred = self.transpose_to_channels_first(y_pred)

        if self.lowpass:  # i.e. lowpass is not None
            """
            apply frequency filter to ground truth data
            note: we are working with tuples, since keras has no complex dtype
            """
            y_true_real, y_true_imag = self.fft(y_true)
            y_true_real, y_true_imag = self.apply_filter(real=y_true_real, imag=y_true_imag)

            y_true, _ = self.ifft((y_true_real, y_true_imag))

        return self.fn(y_true, y_pred, **self._fn_kwargs)
    
    # === everything that has to do with filtering / Fourier domain ===
    def estimate_peak_frequency(self, power_spectrum):
        """
        Estimate peak frequency from spectrum
        
        Parameters
        ----------
        power_spectrum : KerasTensor
            power spectrum of ground truth

        Returns
        -------
        peak_frequency: KerasTensor
            a tensor with shape==(batch_size, ) with peak frequency (capped at self.f_min) for each sample

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
    
    def get_f_hat(self, real, imag):
        """
        Normalize frequency range self.f by estimated peak frequency
        
        Parameters
        ----------
        real : KerasTensor
            real part of Fourier transform of signal y
        imag : KerasTensor
            imaginary part of Fourier transform of signal y

        Returns
        -------
        f_hat : KerasTensor
            frequency range normalized by estimated peak frequency

        """

        ps = self.magnitude(real=real, imag=imag)
        normalized_ps = (ps - ops.min(ps)) / (ops.max(ps) - ops.min(ps))
        
        ps = ops.cast(normalized_ps ** self.p, dtype=real.dtype)
        peak_frequency = self.estimate_peak_frequency(power_spectrum=ps)
        return ops.divide_no_nan(
            ops.expand_dims(self.f, axis=0),  # add batch dimension
            self.expand_dims(peak_frequency, axis=-1)  # add space dimension
        )
    
    def get_frequency_filter(self, real, imag):
        """
        The frequency filter is implemented as a hard binary window, which is multiplied with the Fourier spectrum.
        The window requires
            1. the overall length (int), and 
            2. the length of the window, which is here given by the index where the filter frequency exceeds the frequency vector

        The radius of the filter is calculated based on the frequency grid 'f' and 'f_filter'.
        The closest frequency component to 'f_filter' is found by 'diff = abs(f - f_filter)'.
        The index of the minimum entry in 'diff' is used to calculate the radius of the filter.
        
        All calculations are performed on the positive quadrant of f.
        For the adaptive filter, the frequency spectrum scaled by the peak frequency is used.

        Parameters
        ----------
        real : KerasTensor
            real part of Fourier transform of signal y
        imag : KerasTensor
            imaginary part of Fourier transform of signal y

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

    def lowpass_decay(self, spectral_radius):
        """
        We want to have a linear decrease in spectral radius over the epochs,
        starting from full frequency range to spectral radius.
        The linear function is consequently given by
            s = int(spectral_radius + alpha*(s - spectral_radius))
            with
                s: spectral radius(epoch)
                alpha: slope of linear function, i.e., (epoch - self.decay_start) / self.decay_epochs

        Parameters
        ----------
        spectral_radius : int
            Desired radius of the binary window.
        
        Returns
        -------
        radius : int
            Actual radius at self.epoch
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
    
    def apply_filter(self, real, imag):
        """
        Apply frequency filter
        
        Parameters
        ----------
        real : KerasTensor
            real part of Fourier transform of signal y
        imag : KerasTensor
            imaginary part of Fourier transform of signal y

        Returns
        -------
        [real, imag] : Tuple[KerasTensor]
            Fourier transform of signal y with frequencies > f_filter set to (0 + 0j)
        """
        frequency_filter = self.get_frequency_filter(real, imag)
        return ops.multiply(real, frequency_filter), ops.multiply(imag, frequency_filter)
    
    def lowpass_fn(self, spectral_radius):
        """
        Lowpass function

        Parameters
        ----------
        spectral_radius : int
            Desired radius of the binary window.

        Returns
        -------
        hard_lowpass : KerasTensor
            binary lowpass filter

        """

        # adjust spectral radius
        spectral_radius = self.lowpass_decay(spectral_radius=spectral_radius)
        return hard_lowpass(n=self.nx, spectral_radius=spectral_radius)
    
    def fftshift(self, x, axis=None):
        """
        FFT shift

        shifts the FFT spectra

        Parameters
        ----------
        x : KerasTensor
            signal
        axis : int, optional
            axis to shift along.
            Defaults to None.

        Returns
        -------
        x_shifted : KerasTensor
            fft-shifted version of 'x'

        """

        return ops.roll(x, shift=self.nx // 2, axis=axis)
    
    @staticmethod
    def magnitude(real, imag):
        """
        Magnitude of signal

        Parameters
        ----------
        real : KerasTensor
            real part of Fourier transform of signal y
        imag : KerasTensor
            imaginary part of Fourier transform of signal y

        Returns
        -------
        magnitude : KerasTensor
            magnitude of spectrum

        """

        return ops.sqrt(ops.square(real) + ops.square(imag))
    
    # === serialization ===
    def get_config(self) -> dict:
        base_config: dict = super().get_config()
        config = {
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
        return cls(f=f, **config)
    
    # === helper routines ===
    def transpose_to_channels_first(self, inputs):
        """
        Transpose input data to data format "channels_first"

        The FFT is by default applied along the last dimension of the data.
        Therefore, we have to transpose the data from "channels_last" (default) to "channels_first"

        Parameters
        ----------
        inputs : KerasTensor
            input tensor to transpose

        Returns
        -------
        transposed_inputs : KerasTensor
            input tensor in data format "channels_first"

        """

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
        """
        Casts a KerasTensor to a list

        Required for serialization of a KerasTensor.

        Parameters
        ----------
        arr : KerasTensor
            array to be casted to list

        Returns
        -------
        l : list
            arr as (nested) list

        Notes
        -----
        Raises RuntimeError when 'torch' backend is used.

        """

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
    """
    Base class FrequencyLossFunctionWrapper2D to implement new loss functions with the option to apply a frequency filter to the ground truth.
    This frequency filter helps the model to focus on the relevant frequency range without the need to, e.g., remove HF noise in additional preprocessing steps.

    There are two lowpass filters to choose from, the "static" and the "adaptive" lowpass.
    The "static" lowpass defines a global cut-off frequency at 'f_filter'.
    The "adaptive" lowpass analyzes the ground truth data, extracts the peak frequency, and sets a dynamic cut-off frequency at for each sample.
    The parameter 'f_filter' becomes a multiplier for the peak frequency, after which the frequency components are suppressed.

    The definition of a lowpass {"static", "adaptive"} requires a frequency range 'f'.
    It enables an additional step in the loss calculation, where 
        1. the ground truth is transformed via 2-D FFT,
        2. a hard binary lowpass filter is applied to the Fourier spectrum to set all frequencies f>f_filter to 0+0j,
        3. the filtered ground truth is transformed back to its initial space.
    If lowpass==None, the FFT calculation is skipped, and no 'f' is required.

    This class inherits from keras.losses.Loss and can thus be used directly in model.compile()

    Parameters
    ----------
    fn : callable
        Definition of the loss function.
        The function has to accept two tensors (y_true and y_pred) and return a float.
    lowpass : str, optional {None, "static", "adaptive"}
        Lowpass filter that is applied to the ground truth in order to suppress the higher frequency range 'f>f_filter'.
        Defaults to None.
    f : array_like, optional
        Frequency range for the data.
        Is required once a lowpass {"static", "adaptive"} is used.
        A 1-D f is automatically casted to a 2-D grid.
        Defaults to None.
    f_filter : float, optional
        Threshold for the lowpass filter.
        With the static lowpass, the ground truth spectrum is set to 0+j0 for 'f>f_filter'.
        With the adaptive lowpass, the ground truth spectrum is set to 0+j0 for 'f>f_p * f_filter',
        where f_p is the peak frequency that is automatically derived from the ground truth spectrum.
        Defaults to 6.0.
    f_min : float, optional
        Cap for the lowest peak frequency for cases when the automatic estimation of the peak frequency fails (estimated 'f_p < 0' or 'f_p == NaN').
        Defaults to 0.0.
    p : float, optional
        Exponent to weigh the spectrum towards the peak frequency (for the estimation of the peak frequency), c.f. 
            Mansard & Funke, "On the fitting of parametric models to measured wave spectra" (1988), and
            Sobey & Young, "Hurricane Wind Waves---A discrete spectral model" (1986), https://ascelibrary.org/doi/10.1061/%28ASCE%290733-950X%281986%29112%3A3%28370%29
        Defaults to 7.0.
    decay_start : int, optional !!!Requires 'UseLossLowpassDecay' callback to work!!!
        Epoch from which on the lowpass filter is linearly decreased from 0 to 'f_filter'.
        Defaults to 0.
    decay_epochs : int, optional !!!Requires 'UseLossLowpassDecay' callback to work!!!
        Number of epochs over which the lowpass filter is linearly decreased from 0 to 'f_filter'.
        Defaults to 50.
    data_format : str, optional {"channels_last", "channels_first"}
        The ordering of the dimensions in the inputs:
        "channels_last" corresponds to inputs with shape (batch_size, *dims, channels), 
        "channels_first" corresponds to inputs with shape (batch_size, channels, *dims).
        Defaults to "channels_last".
    reduction : str, optional {"sum_over_batch_size", "None", "auto", "sum"}
        Type of reduction to apply to the loss.
        In almost all cases this should be `"sum_over_batch_size"`.
        Supported options are `"sum"`, `"sum_over_batch_size"` or `None`.
    name : str, optional
        Name of the loss function. Name is inhereted from class name if name=None.
        Defaults to None.
    **kwargs
        Additional keyword arguments for fn.

    Notes
    -----
    Both the "adaptive" and "static" lowpass filter can be linearly increased from 0 to f_filter over 'decay_epoch' epochs, starting at epoch 'decay_start'.
    For this to work, the training has to be conducted using the 'UseLossLowpassDecay' callback, which sets the class variable 'self.epoch' to the current training epoch.
    See examples of SSP1D and SSP2D for a MWE.

    """
    
    def __init__(
        self,
        fn,
        lowpass=None,
        f=None,
        f_filter=6.0,
        f_min=0.0,
        p=7.0,
        data_format="channels_last",
        reduction="sum_over_batch_size",
        name=None,
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
    def get_frequency_filter(self, real, imag):
        """
        The frequency filter is implemented as a hard binary window, which is multiplied with the Fourier spectrum.
        The window requires
            1. the overall length (int), and 
            2. the length of the window, which is here given by the index where the filter frequency exceeds the frequency vector

        The radius of the filter is calculated based on the frequency grid 'f' and 'f_filter'.
        The closest frequency component to 'f_filter' is found by 'diff = abs(f - f_filter)'.
        The index of the minimum entry in 'diff' is used to calculate the radius of the filter.
        
        All calculations are performed on the positive quadrant of f.
        For the adaptive filter, the frequency spectrum scaled by the peak frequency is used.

        Parameters
        ----------
        real : KerasTensor
            real part of Fourier transform of signal y
        imag : KerasTensor
            imaginary part of Fourier transform of signal y

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
        
    # def get_config(self) -> dict:
    #     """
    #     while 2D loss does not support LowpassDecay, this function has to be implemented without arguments
    #         decay_epochs
    #         decay_start
    #     """
    #     config: dict = super().get_config()
    #     config.pop("decay_epochs")
    #     config.pop("decay_start")

    #     return config
    
    # === helper routines adjusted for 2D ===
    def transpose_to_channels_first(self, inputs):
        """
        Transpose input data to data format "channels_first"

        The FFT is by default applied along the last dimension of the data.
        Therefore, we have to transpose the data from "channels_last" (default) to "channels_first"

        Parameters
        ----------
        inputs : KerasTensor
            input tensor to transpose

        Returns
        -------
        transposed_inputs : KerasTensor
            input tensor in data format "channels_first"

        """

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

        Parameters
        ----------
        spectral_radius : int
            Desired radius of the binary window.
        
        Returns
        -------
        radius : int
            Actual radius at self.epoch
        """
        
        if self.epoch is None:
            # callback to steer decay is not active, just return spectral radius
            return spectral_radius
        
        if self.epoch > (self.decay_start + self.decay_epochs):
            return spectral_radius
        
        max_radius = ops.cast(ops.round(self.magnitude(self.nx // 2 - 1, self.ny // 2 - 1)), dtype='int32')  # zero indexing, we need positive frequencies only

        if self.epoch < self.decay_start:
            return max_radius
        
        alpha = ops.divide(self.epoch - self.decay_start, self.decay_epochs)
        return ops.cast(ops.cast(max_radius, dtype=alpha.dtype) + alpha * ops.cast(spectral_radius - max_radius, dtype=alpha.dtype), dtype=spectral_radius.dtype)
    
    def lowpass_fn(self, spectral_radius):
        """
        Lowpass function

        Parameters
        ----------
        spectral_radius : int
            Desired radius of the binary window.

        Returns
        -------
        hard_lowpass : KerasTensor
            binary lowpass filter

        """

        spectral_radius = self.lowpass_decay(spectral_radius=spectral_radius)
        return circular_hard_lowpass(n=self.nx, spectral_radius=spectral_radius)
    
    def fftshift(self, x):
        """
        FFT shift

        shifts the FFT spectra along the last 2 axes

        Parameters
        ----------
        x : KerasTensor
            signal

        Returns
        -------
        x_shifted : KerasTensor
            fft-shifted version of 'x'

        """

        return ops.roll(ops.roll(x, shift=self.ny // 2, axis=-2), shift=self.nx // 2, axis=-1)
