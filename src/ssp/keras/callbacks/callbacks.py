from keras import callbacks
from keras.src.utils import io_utils


class UseLossLowpassDecay(callbacks.Callback):
    """
    UseLossLowpassDecay Callback.
    This callback updates the class variable `loss.epoch` to the current training epoch in order to allow a linear decay of the lowpass filter.

    """

    def __init__(self):
        super().__init__()

        self.supported_loss = True

    def on_train_begin(self, logs=None):
        """
        Check if loss function is supported and set `loss.epoch=0`

        """

        try:
            # set loss.epoch from None to 0
            self.model.loss.epoch = 0
        except AttributeError:
            io_utils.print_msg(f"UseLossLowpassDecay: unsupported loss {self.model.loss.name}.")
            self.supported_loss = False

    def on_epoch_begin(self, epoch, logs=None):
        """
        Update `loss.epoch` to the current training epoch

        """

        if self.supported_loss:
            self.model.loss.epoch = epoch
