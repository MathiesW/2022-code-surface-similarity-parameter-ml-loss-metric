from .base_frequency_loss import FrequencyLossFunctionWrapper1D, FrequencyLossFunctionWrapper2D, squeeze_or_expand_to_same_rank
from keras import KerasTensor
from keras import ops
from keras import saving


@saving.register_keras_serializable(package="CustomLosses", name="normalized_error")
def normalized_error(y_true, y_pred):
    """
    Normalized error (Perlin & Bustamante, cf. https://doi.org/10.1007/s10665-016-9849-7)

    Definition of the Surface Similarity Parameter (SSP) for discrete signals
        SSP = norm(y_true - y_pred) / (norm(y_true) + norm(y_pred))

    Parameters
    ----------
    y_true : KerasTensor
        Ground truth signal
    y_pred : KerasTensor
        Predicted signal

    Returns
    -------
    ssp : KerasTensor
        normalized error (SSP) between 'y_true' and 'y_pred'
        
    """

    numerator = ops.linalg.norm(y_true - y_pred, axis=-1)
    denuminator = ops.add(
            ops.linalg.norm(y_true, axis=-1),
            ops.linalg.norm(y_pred, axis=-1)
        )
    
    return ops.divide_no_nan(
        numerator,
        denuminator
    )


@saving.register_keras_serializable(package="CustomLosses", name="SSP1D")
class SSP1D(FrequencyLossFunctionWrapper1D):
    """
    Surface Similarity Parameter for 1-D signals with an optional lowpass filter.
    The frequency filter helps the model to focus on the relevant frequency range without the need to, e.g., remove HF noise in additional preprocessing steps.

    There are two lowpass filters to choose from, the `"static"` and the `"adaptive"` lowpass.
    The `"static"` lowpass defines a global cut-off frequency at `f_filter`.
    The `"adaptive"` lowpass analyzes the ground truth data, extracts the peak frequency, and sets a dynamic cut-off frequency at for each sample.
    The parameter `f_filter` becomes a multiplier for the peak frequency, after which the frequency components are suppressed.

    The definition of a lowpass {`"static"`, `"adaptive"`} requires a frequency range `f`.
    It enables an additional step in the loss calculation, where
    (1) the ground truth is transformed via 1-D FFT, 
    (2) a hard binary lowpass filter is applied to the Fourier spectrum to set all frequencies `f>f_filter` to (0+0j), 
    (3) the filtered ground truth is transformed back to its initial space.

    If `lowpass==None`, the FFT calculation is skipped, and no `f` is required.

    This class inherits from keras.losses.Loss and can thus be used directly in keras.Model.compile()

    Parameters
    ----------
    lowpass : str, optional {`None`, `"static"`, `"adaptive"`}
        Lowpass filter that is applied to the ground truth in order to suppress the higher frequency range `f>f_filter`.
        Defaults to `None`.
    f : KerasTensor, optional
        Frequency range for the data.
        Is required once a lowpass {`"static"`, `"adaptive"`} is used.
        Defaults to `None`.
    f_filter : float, optional
        Threshold for the lowpass filter.
        With the static lowpass, the ground truth spectrum is set to 0+j0 for `f>f_filter`.
        With the adaptive lowpass, the ground truth spectrum is set to 0+j0 for `f>f_filter*f_p`,
        where `f_p` is the peak frequency that is automatically derived from the ground truth spectrum.
        Defaults to 6.0.
    f_min : float, optional
        Cap for the lowest peak frequency for cases when the automatic estimation of the peak frequency fails (estimated `f_p<0` or `f_p` is Nan).
        Defaults to 0.0.
    p : float, optional
        Exponent to weigh the spectrum towards the peak frequency (for the estimation of the peak frequency), c.f. 
        Mansard & Funke, "On the fitting of parametric models to measured wave spectra" (1988), and
        Sobey & Young, "Hurricane Wind Waves---A discrete spectral model" (1986), https://ascelibrary.org/doi/10.1061/%28ASCE%290733-950X%281986%29112%3A3%28370%29.
        Defaults to 7.0.
    decay_start : int, optional
        Epoch from which on the lowpass filter is linearly decreased from 0 to `f_filter`.
        Defaults to 0.
        **Requires `UseLossLowpassDecay` callback to work, cf. Notes**
    decay_epochs : int, optional
        Number of epochs over which the lowpass filter is linearly decreased from 0 to `f_filter`.
        Defaults to 50.
        **Requires `UseLossLowpassDecay` callback to work, cf. Notes**
    data_format : str, optional {`"channels_last"`, `"channels_first"`}
        The ordering of the dimensions in the inputs:
        `"channels_last"` corresponds to inputs with shape `(batch_size, *dims, channels)`, 
        `"channels_first"` corresponds to inputs with shape `(batch_size, channels, *dims)`.
        Defaults to `"channels_last"`.
    reduction : str, optional {`"sum_over_batch_size"`, `None`, `"auto"`, `"sum"`}
        Type of reduction to apply to the loss.
        In almost all cases this should be `"sum_over_batch_size"`.
        Supported options are `"sum"`, `"sum_over_batch_size"` or `None`.
    name : str, optional
        Name of the loss function. The name is inhereted from class name if `name=None`.
        Defaults to `None`.

    Notes
    -----
    Both the `"adaptive"` and `"static"` lowpass filter can be linearly increased from 0 to `f_filter` over `decay_epoch` epochs, starting at epoch `decay_start`.
    For this to work, the training has to be conducted using the `UseLossLowpassDecay` callback, which sets the class variable `self.epoch` to the current training epoch.

    Examples
    --------
    >>> from ssp.keras import SSP1D
    >>> from keras import ops
    >>> from math import pi
    >>> t = ops.arange(0, 2 * pi, 2 * pi / 512)
    >>> y1 = ops.sin(t)
    >>> y2 = ops.sin(t + pi / 16)  # sin with phase shift
    >>> ssp = SSP1D()
    >>> ops.convert_to_numpy(ssp(y1, y2))
    array(0.09801717, dtype=float32)

    
    Usage in an ML training
    
    >>> from ssp.keras import SSP1D
    >>> from keras import ops, Sequential, layers
    >>> from math import pi
    >>> t = ops.arange(0, 2 * pi, 2 * pi / 512)
    >>> y = ops.expand_dims(ops.sin(t), axis=0)  # shape (1, 512)
    >>> x = ops.ones((1, 32))  # some input data with shape (1, 32)
    >>> model = Sequential([layers.Dense(64), layers.Dense(512)])
    >>> model.build(input_shape=x.shape)
    >>> model.compile(optimizer="adam", loss=SSP1D())
    >>> model.fit(x=x, y=y, epochs=1)

    
    Usage in an ML training with static lowpass *(note that the lowpass does not make sense since we have a harmonic sine as input)*

    >>> from ssp.keras import SSP1D
    >>> from ssp.keras.ops import fftfreq
    >>> from keras import ops, Sequential, layers
    >>> from math import pi
    >>> t = ops.arange(0, 2 * pi, 2 * pi / 512)
    >>> f = fftfreq(n=512, d=2 * pi / 512)
    >>> y = ops.expand_dims(ops.sin(t), axis=0)  # shape (1, 512)
    >>> x = ops.ones((1, 32))  # some input data with shape (1, 32)
    >>> model = Sequential([layers.Dense(64), layers.Dense(512)])
    >>> model.build(input_shape=x.shape)
    >>> model.compile(optimizer="adam", loss=SSP1D(lowpass="static", f=f, f_filter=2.0))
    >>> model.fit(x=x, y=y, epochs=1)

    
    Usage in an ML training with adaptive lowpass *(note that the lowpass does not make sense since we have a harmonic sine as input)*

    >>> from ssp.keras import SSP1D
    >>> from ssp.keras.ops import fftfreq
    >>> from keras import ops, Sequential, layers
    >>> from math import pi
    >>> t = ops.arange(0, 2 * pi, 2 * pi / 512)
    >>> f = fftfreq(n=512, d=2 * pi / 512)
    >>> y = ops.expand_dims(ops.sin(t), axis=0)  # shape (1, 512)
    >>> x = ops.ones((1, 32))  # some input data with shape (1, 32)
    >>> model = Sequential([layers.Dense(64), layers.Dense(512)])
    >>> model.build(input_shape=x.shape)
    >>> model.compile(optimizer="adam", loss=SSP1D(lowpass="adaptive", f=f, f_filter=6.0))
    >>> model.fit(x=x, y=y, epochs=1)

    
    Usage in an ML training with adaptive lowpass and decay callback *(note that the lowpass does not make sense since we have a harmonic sine as y)*

    >>> from ssp.keras import SSP1D
    >>> from ssp.keras.callbacks import UseLossLowpassDecay
    >>> from ssp.keras.ops import fftfreq
    >>> from keras import ops, Sequential, layers
    >>> from math import pi
    >>> t = ops.arange(0, 2 * pi, 2 * pi / 512)
    >>> f = fftfreq(n=512, d=2 * pi / 512)
    >>> y = ops.expand_dims(ops.sin(t), axis=0)  # shape (1, 512)
    >>> x = ops.ones((1, 32))  # some input data with shape (1, 32)
    >>> model = Sequential([layers.Dense(64), layers.Dense(512)])
    >>> model.build(input_shape=x.shape)
    >>> model.compile(optimizer="adam", loss=SSP1D(lowpass="adaptive", f=f, f_filter=6.0, decay_start=0, decay_epochs=5))
    >>> model.fit(x=x, y=y, epochs=10, callbacks=[UseLossLowpassDecay()])

    """

    def __init__(
        self, 
        lowpass=None, 
        f=None, 
        f_filter=6, 
        f_min=0, 
        p=7, 
        decay_epochs=50,
        decay_start=0,
        data_format="channels_last", 
        reduction="sum_over_batch_size", 
        name=None
    ):
        super().__init__(
            fn=normalized_error, 
            lowpass=lowpass, 
            f=f, 
            f_filter=f_filter, 
            f_min=f_min, 
            p=p, 
            decay_epochs=decay_epochs,
            decay_start=decay_start,
            data_format=data_format, 
            reduction=reduction, 
            name=name
        )

    def call(self, y_true, y_pred):
        """
        Call method of SSP1D

        Parameters
        ----------
        y_true : KerasTensor
            Ground truth signal
        y_pred : KerasTensor
            Predicted signal

        Returns
        -------
        y : KerasTensor
            Surface Similarity Parameter between arrays `y_true` and `y_pred`.
            The shape is determined by `self.reduction`.

        Notes
        -----
        The call method of `FrequencyLossFunctionWrapper1D` is overwritten, since with the SSP, we can remain in frequency domain and omit the ifft. 
        Magically, it is the same result if the SSP is applied in time- or frequency domain!

        """

        y_true = ops.convert_to_tensor(y_true)  # shape = (batch, x, ch) if data_format == 'channels_last' else (batch, x) or (batch, 1, x)
        y_pred = ops.convert_to_tensor(y_pred)

        # squeeze along channel axis
        y_true, y_pred = squeeze_or_expand_to_same_rank(y_true, y_pred, axis=-1 if self.data_format == 'channels_last' else 1)  # this adds channel dimension to y_true if there is none

        # make channels first for FFT application
        y_true = self.transpose_to_channels_first(y_true)  # shape = (batch, ch, x) or (batch, x)
        y_pred = self.transpose_to_channels_first(y_pred)
        
        if self.lowpass:  # i.e. lowpass is not None
            # we are working with tuples, since keras has no complex dtype
            y_true_real, y_true_imag = self.fft(y_true)
            y_pred_real, y_pred_imag = self.fft(y_pred)

            y_true_real, y_true_imag = self.apply_filter(real=y_true_real, imag=y_true_imag)

            # the ssp loss is taking the L2 norm of the 
            y_true = ops.concatenate((y_true_real, y_true_imag), axis=1)
            y_pred = ops.concatenate((y_pred_real, y_pred_imag), axis=1)

        return self.fn(y_true=y_true, y_pred=y_pred)


@saving.register_keras_serializable(package="CustomLosses", name="SSP2D")
class SSP2D(FrequencyLossFunctionWrapper2D):
    """
    Surface Similarity Parameter for 2-D signals with an optional lowpass filter.
    The frequency filter helps the model to focus on the relevant frequency range without the need to, e.g., remove HF noise in additional preprocessing steps.

    There are two lowpass filters to choose from, the `"static"` and the `"adaptive"` lowpass.
    The `"static"` lowpass defines a global cut-off frequency at `f_filter`.
    The `"adaptive"` lowpass analyzes the ground truth data, extracts the peak frequency, and sets a dynamic cut-off frequency at for each sample.
    The parameter `f_filter` becomes a multiplier for the peak frequency, after which the frequency components are suppressed.

    The definition of a lowpass {`"static"`, `"adaptive"`} requires a frequency range `f`.
    It enables an additional step in the loss calculation, where
    (1) the ground truth is transformed via 1-D FFT, 
    (2) a hard binary lowpass filter is applied to the Fourier spectrum to set all frequencies `f>f_filter` to (0+0j), 
    (3) the filtered ground truth is transformed back to its initial space.

    If `lowpass==None`, the FFT calculation is skipped, and no `f` is required.

    This class inherits from keras.losses.Loss and can thus be used directly in keras.Model.compile()

    Parameters
    ----------
    lowpass : str, optional {`None`, `"static"`, `"adaptive"`}
        Lowpass filter that is applied to the ground truth in order to suppress the higher frequency range `f>f_filter`.
        Defaults to `None`.
    f : KerasTensor, optional
        Frequency range for the data.
        Is required once a lowpass {`"static"`, `"adaptive"`} is used.
        A 1-D `f` is automatically casted to a 2-D grid.
        Defaults to `None`.
    f_filter : float, optional
        Threshold for the lowpass filter.
        With the static lowpass, the ground truth spectrum is set to 0+j0 for `f>f_filter`.
        With the adaptive lowpass, the ground truth spectrum is set to 0+j0 for `f>f_filter*f_p`,
        where `f_p` is the peak frequency that is automatically derived from the ground truth spectrum.
        Defaults to 6.0.
    f_min : float, optional
        Cap for the lowest peak frequency for cases when the automatic estimation of the peak frequency fails (estimated `f_p<0` or `f_p` is Nan).
        Defaults to 0.0.
    p : float, optional
        Exponent to weigh the spectrum towards the peak frequency (for the estimation of the peak frequency), c.f. 
        Mansard & Funke, "On the fitting of parametric models to measured wave spectra" (1988), and
        Sobey & Young, "Hurricane Wind Waves---A discrete spectral model" (1986), https://ascelibrary.org/doi/10.1061/%28ASCE%290733-950X%281986%29112%3A3%28370%29.
        Defaults to 7.0.
    decay_start : int, optional
        Epoch from which on the lowpass filter is linearly decreased from 0 to `f_filter`.
        Defaults to 0.
        **Requires `UseLossLowpassDecay` callback to work, cf. Notes**
    decay_epochs : int, optional
        Number of epochs over which the lowpass filter is linearly decreased from 0 to `f_filter`.
        Defaults to 50.
        **Requires `UseLossLowpassDecay` callback to work, cf. Notes**
    data_format : str, optional {`"channels_last"`, `"channels_first"`}
        The ordering of the dimensions in the inputs:
        `"channels_last"` corresponds to inputs with shape `(batch_size, *dims, channels)`, 
        `"channels_first"` corresponds to inputs with shape `(batch_size, channels, *dims)`.
        Defaults to `"channels_last"`.
    reduction : str, optional {`"sum_over_batch_size"`, `None`, `"auto"`, `"sum"`}
        Type of reduction to apply to the loss.
        In almost all cases this should be `"sum_over_batch_size"`.
        Supported options are `"sum"`, `"sum_over_batch_size"` or `None`.
    name : str, optional
        Name of the loss function. The name is inhereted from class name if `name=None`.
        Defaults to `None`.

    Notes
    -----
    Both the `"adaptive"` and `"static"` lowpass filter can be linearly increased from 0 to `f_filter` over `decay_epoch` epochs, starting at epoch `decay_start`.
    For this to work, the training has to be conducted using the `UseLossLowpassDecay` callback, which sets the class variable `self.epoch` to the current training epoch.

    Examples
    --------
    >>> from ssp.keras import SSP2D
    >>> from keras import ops
    >>> from math import pi
    >>> x = ops.arange(-2 * pi, 2 * pi, 4 * pi / 512)
    >>> xx, yy = ops.meshgrid(x, x)  # just a simple square domain from -2pi to 2pi in x and y
    >>> y1 = ops.sin(xx) + ops.cos(yy)
    >>> y2 = ops.sin(xx + pi / 16) + ops.cos(yy + pi / 16)  # sin with phase shift in x and y direction
    >>> ssp = SSP2D()
    >>> ops.convert_to_numpy(ssp(y1, y2))
    array(0.10495476, dtype=float32)

    
    Usage in an ML training

    >>> from ssp.keras import SSP2D
    >>> from keras import ops, Sequential, layers
    >>> from math import pi
    >>> x = ops.arange(-2 * pi, 2 * pi, 4 * pi / 512)
    >>> xx, yy = ops.meshgrid(x, x)  # just a simple square domain from -2pi to 2pi in x and y
    >>> y = ops.sin(xx) + ops.cos(yy)
    >>> y = ops.expand_dims(y, axis=0)
    >>> inputs = ops.ones((1, 512, 512))
    >>> model = Sequential([layers.Conv2D(filters=1, kernel_size=3, padding="same"), layers.Reshape(target_shape=(512, 512))])
    >>> model.build(input_shape=inputs.shape)
    >>> model.compile(optimizer="adam", loss=SSP2D())
    >>> model.fit(x=inputs, y=y, epochs=1)

    
    Usage in an ML training with static lowpass

    >>> from ssp.keras import SSP2D
    >>> from ssp.keras.ops import fftfreq
    >>> from keras import ops, Sequential, layers
    >>> from math import pi
    >>> x = ops.arange(-2 * pi, 2 * pi, 4 * pi / 512)
    >>> xx, yy = ops.meshgrid(x, x)  # just a simple square domain from -2pi to 2pi in x and y
    >>> y = ops.sin(xx) + ops.cos(yy)
    >>> y = ops.expand_dims(y, axis=0)
    >>> inputs = ops.ones((1, 512, 512))
    >>> model = Sequential([layers.Conv2D(filters=1, kernel_size=3, padding="same"), layers.Reshape(target_shape=(512, 512))])
    >>> model.build(input_shape=inputs.shape)
    >>> model.compile(optimizer="adam", loss=SSP2D(lowpass="static", f=f, f_filter=2.0))
    >>> model.fit(x=inputs, y=y, epochs=1)

    
    Usage in an ML training with adaptive lowpass

    >>> from ssp.keras import SSP2D
    >>> from ssp.keras.ops import fftfreq
    >>> from keras import ops, Sequential, layers
    >>> from math import pi
    >>> x = ops.arange(-2 * pi, 2 * pi, 4 * pi / 512)
    >>> xx, yy = ops.meshgrid(x, x)  # just a simple square domain from -2pi to 2pi in x and y
    >>> y = ops.sin(xx) + ops.cos(yy)
    >>> y = ops.expand_dims(y, axis=0)
    >>> inputs = ops.ones((1, 512, 512, 1))
    >>> model = Sequential([layers.Conv2D(filters=1, kernel_size=3, padding="same"), layers.Reshape(target_shape=(512, 512))])
    >>> model.build(input_shape=x.shape)
    >>> model.compile(optimizer="adam", loss=SSP2D(lowpass="adaptive", f=f, f_filter=6.0))
    >>> model.fit(x=inputs, y=y, epochs=1)

    
    Usage in an ML training with adaptive lowpass and decay callback

    >>> from src.ssp.keras import SSP2D
    >>> from src.ssp.keras.callbacks import UseLossLowpassDecay
    >>> from src.ssp.keras.ops import fftfreq
    >>> from keras import ops, Sequential, layers
    >>> from math import pi
    >>> x = ops.arange(-2 * pi, 2 * pi, 4 * pi / 512)
    >>> k = fftfreq(n=512, d=4 * pi / 512, rad=True)
    >>> xx, yy = ops.meshgrid(x, x)  # just a simple square domain from -2pi to 2pi in x and y
    >>> y = ops.sin(xx) + ops.cos(yy)
    >>> y = ops.expand_dims(y, axis=0)
    >>> inputs = ops.ones((1, 512, 512, 1))  # input has to have a channel dimension with data format "channels_last"
    >>> model = Sequential([layers.Conv2D(filters=1, kernel_size=3, padding="same"), layers.Reshape(target_shape=(512, 512))])
    >>> model.build(input_shape=inputs.shape)
    >>> model.compile(optimizer="adam", loss=SSP2D(lowpass="adaptive", f=k, f_filter=6.0, decay_start=0, decay_epochs=5))
    >>> model.fit(x=inputs, y=y, epochs=10, callbacks=[UseLossLowpassDecay()])

    """

    def __init__(
        self, 
        lowpass=None, 
        f=None, 
        f_filter=6, 
        f_min=0, 
        p=7, 
        decay_epochs=50,
        decay_start=0,
        data_format="channels_last", 
        reduction="sum_over_batch_size", 
        name=None
    ):
        super().__init__(
            fn=normalized_error, 
            lowpass=lowpass, 
            f=f, 
            f_filter=f_filter, 
            f_min=f_min, 
            p=p, 
            decay_epochs=decay_epochs,
            decay_start=decay_start,
            data_format=data_format, 
            reduction=reduction, 
            name=name
        )

    def call(self, y_true, y_pred):
        """
        Call method of SSP2D

        Parameters
        ----------
        y_true : KerasTensor
            Ground truth signal
        y_pred : KerasTensor
            Predicted signal

        Returns
        -------
        y : KerasTensor
            Surface Similarity Parameter between arrays `y_true` and `y_pred`.
            The shape is determined by `self.reduction`.

        Notes
        -----
        The call method of `FrequencyLossFunctionWrapper2D` is overwritten, since with the SSP, we can remain in frequency domain and omit the ifft. 
        Magically, it is the same result if the SSP is applied in time- or frequency domain!

        The 2-D implementation of the SSP uses the same `normalized_error` function as the 1-D implementation.
        This is made possible by flattening the Fourier-transformed arrays along all dimensions that are not the batch dimension.
        Then, the norm along the feature axis is taken to calculate the SSP2D.

        """

        y_true = ops.convert_to_tensor(y_true)  # shape = (batch, x, ch) if data_format == 'channels_last' else (batch, x) or (batch, 1, x)
        y_pred = ops.convert_to_tensor(y_pred)

        # squeeze along channel axis
        y_true, y_pred = squeeze_or_expand_to_same_rank(y_true, y_pred, axis=-1 if self.data_format == 'channels_last' else 1)  # this adds channel dimension to y_true if there is none

        # make channels first for FFT application
        y_true = self.transpose_to_channels_first(y_true)  # shape = (batch, ch, x) or (batch, x)
        y_pred = self.transpose_to_channels_first(y_pred)

        # derive batch size for reshape (flatten) later
        b = ops.shape(y_true)[0]
        
        if self.lowpass:  # i.e. lowpass is not None
            y_true_real, y_true_imag = self.fft(y_true)
            y_pred_real, y_pred_imag = self.fft(y_pred)

            y_true_real, y_true_imag = self.apply_filter(real=y_true_real, imag=y_true_imag)

            # the ssp loss is taking the L2 norm of the flattened spectra
            # 1st, reshape all arrays, i.e., flatten along feature axes
            y_true_real = ops.reshape(y_true_real, newshape=(b, -1))
            y_true_imag = ops.reshape(y_true_imag, newshape=(b, -1))
            y_pred_real = ops.reshape(y_pred_real, newshape=(b, -1))
            y_pred_imag = ops.reshape(y_pred_imag, newshape=(b, -1))

            y_true = ops.concatenate((y_true_real, y_true_imag), axis=1)
            y_pred = ops.concatenate((y_pred_real, y_pred_imag), axis=1)

        else:
            # flatten arrays
            y_true = ops.reshape(y_true, newshape=(b, -1))
            y_pred = ops.reshape(y_pred, newshape=(b, -1))

        return self.fn(y_true=y_true, y_pred=y_pred)
