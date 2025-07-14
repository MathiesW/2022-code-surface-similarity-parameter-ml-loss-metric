from .base_frequency_loss import FrequencyLossFunctionWrapper1D, FrequencyLossFunctionWrapper2D, squeeze_or_expand_to_same_rank
from keras import KerasTensor
from keras import ops
from keras import saving


@saving.register_keras_serializable(package="CustomLosses", name="normalized_error")
def normalized_error(y_true: KerasTensor, y_pred: KerasTensor) -> KerasTensor:
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
        y_true = ops.convert_to_tensor(y_true)  # shape = (batch, x, ch) if data_format == 'channels_last' else (batch, x) or (batch, 1, x)
        y_pred = ops.convert_to_tensor(y_pred)

        # squeeze along channel axis
        y_true, y_pred = squeeze_or_expand_to_same_rank(y_true, y_pred, axis=-1 if self.data_format == 'channels_last' else 1)  # this adds channel dimension to y_true if there is none

        # make channels first for FFT application
        y_true = self.transpose_to_channels_first(y_true)  # shape = (batch, ch, x) or (batch, x)
        y_pred = self.transpose_to_channels_first(y_pred)
        
        if self.lowpass:  # i.e. lowpass is not None
            """ apply frequency filter to ground truth data
            NOTE: with SSP, we remain in frequency domain and omit the ifft. 
                Magically, it is the same result if the SSP is applied in time- or frequency domain!
                However, when one of the signals is filtered, it may not be the same, so better stay in frequency domain
            NOTE: we are working with tuples, since keras has no complex dtype
            """
            y_true_real, y_true_imag = self.fft(y_true)
            y_pred_real, y_pred_imag = self.fft(y_pred)

            y_true_real, y_true_imag = self.apply_filter(real=y_true_real, imag=y_true_imag)

            # the ssp loss is taking the L2 norm of the 
            y_true = ops.concatenate((y_true_real, y_true_imag), axis=1)
            y_pred = ops.concatenate((y_pred_real, y_pred_imag), axis=1)

        return self.fn(y_true=y_true, y_pred=y_pred)


@saving.register_keras_serializable(package="CustomLosses", name="SSP2D")
class SSP2D(FrequencyLossFunctionWrapper2D):
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
            """ apply frequency filter to ground truth data
            NOTE: with SSP, we remain in frequency domain and omit the ifft. 
                Magically, it is the same result if the SSP is applied in time- or frequency domain!
                However, when one of the signals is filtered, it may not be the same, so better stay in frequency domain
            NOTE: we are working with tuples, since keras has no complex dtype
            """
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
