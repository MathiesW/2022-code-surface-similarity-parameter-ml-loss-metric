try:
    import keras
except ImportError as e:
    raise ImportError(
        "The 'ssp.keras'-module requires the keras package to work. Install the ssp package with 'pip install ssp[keras]' to automatically install keras>=3.0.0."
    ) from e

from .loss_functions import SSP1D, SSP2D
