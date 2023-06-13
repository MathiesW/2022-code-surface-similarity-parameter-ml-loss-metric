import torch
from torch import nn
import numpy as np
from typing import Union


class _SurfaceSimilarityParameterBase(nn.Module):
    def __init__(
            self,
            dimension: int,
            **kwargs
    ):
        assert dimension in [1, 2]

        super().__init__()
        self.dimension = dimension
        self.fft = torch.fft.fft if dimension == 1 else torch.fft.fft2

    def sobolev_norm(self, y_f: torch.tensor) -> torch.tensor:
        y_f = torch.square(torch.abs(y_f))

        for _ in range(self.dimension):
            y_f = torch.trapz(y_f)

        return torch.sqrt(y_f)

    def forward(self, y_true: torch.tensor, y_pred: torch.tensor) -> torch.tensor:
        assert y_true.shape == y_pred.shape, f'Shape of tensors do not match: {y_true.shape} vs {y_pred.shape}'

        y_true_f = self.fft(y_true)
        y_pred_f = self.fft(y_pred)

        return torch.divide(
            self.sobolev_norm(torch.subtract(y_true_f, y_pred_f)),
            torch.add(self.sobolev_norm(y_true_f), self.sobolev_norm(y_pred_f))
        )


class _SurfaceSimilarityParameterLowPassBase(_SurfaceSimilarityParameterBase):
    def __init__(
            self,
            dimension: int,
            k: Union[np.ndarray, torch.tensor],
            k_filter: float = None,
            lowpass: str = 'adaptive',
            p: float = 8.0,
            **kwargs
    ):
        assert lowpass in ['static', 'adaptive']
        super().__init__(dimension=dimension)

        self.k = torch.tensor(k).type(torch.float32)
        self.p = torch.tensor(p).type(torch.float32)

        if lowpass == 'static':
            self.k_filter = k_filter or 2.0
            self.static_filter = self.get_static_filter()

        if lowpass == 'adaptive':
            self.k_filter = k_filter or 6.0
            self.static_filter = None

    def get_adaptive_filter(self, y_f: torch.tensor) -> torch.tensor:
        return NotImplemented

    def get_static_filter(self) -> torch.tensor:
        return NotImplemented

    def forward(self, y_true: torch.tensor, y_pred: torch.tensor) -> torch.tensor:
        assert y_true.shape == y_pred.shape, f'Shape of tensors do not match: {y_true.shape} vs {y_pred.shape}'

        # add batch dimension if called with unbatched tensors
        y_true_f = self.fft(y_true.unsqueeze(dim=0) if y_true.ndim == self.dimension else y_true)
        y_pred_f = self.fft(y_pred.unsqueeze(dim=0) if y_pred.ndim == self.dimension else y_pred)

        y_true_f *= self.get_adaptive_filter(y_true_f) if self.static_filter is None else self.static_filter

        return torch.divide(
            self.sobolev_norm(torch.subtract(y_true_f, y_pred_f)),
            torch.add(self.sobolev_norm(y_true_f), self.sobolev_norm(y_pred_f))
        )


class SurfaceSimilarityParameter(_SurfaceSimilarityParameterBase):
    def __init__(self, **kwargs):
        """
        Surface similarity parameter (SSP) for one-dimensional data.
        The SSP expects the data format NW with no channel dimension
        """
        super().__init__(dimension=1)


class SurfaceSimilarityParameterLowPass(_SurfaceSimilarityParameterLowPassBase):
    def __init__(
            self,
            k: Union[np.ndarray, torch.tensor],
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
        :param kwargs: unused
        """
        super().__init__(dimension=1, k=k, k_filter=k_filter, lowpass=lowpass, p=p)

    def get_adaptive_filter(self, y_f: torch.tensor) -> torch.tensor:
        spec = torch.abs(y_f) ** self.p
        kp = torch.divide(torch.trapz(torch.abs(self.k * spec)), torch.trapz(spec))

        k_hat = torch.abs(torch.divide(self.k[None, :], kp[:, None]))

        return torch.where(torch.greater_equal(k_hat, self.k_filter), 0, 1).type(torch.complex64)

    def get_static_filter(self) -> torch.tensor:
        return torch.where(torch.greater_equal(torch.abs(self.k), self.k_filter), 0, 1).type(torch.complex64)


class SurfaceSimilarityParameter2D(_SurfaceSimilarityParameterBase):
    def __init__(self, **kwargs):
        """
        Surface similarity parameter (SSP) for two-dimensional data.
        The SSP expects the data format NW with no channel dimension
        """
        super().__init__(dimension=2)


class SurfaceSimilarityParameter2DLowPass(_SurfaceSimilarityParameterLowPassBase):
    def __init__(
            self,
            k: Union[np.ndarray, torch.tensor],
            k_filter: float = None,
            lowpass: str = 'adaptive',
            p: float = 8.0
    ):
        super().__init__(dimension=2, k=k, k_filter=k_filter, lowpass=lowpass, p=p)
        self.grid = torch.sqrt(sum(torch.meshgrid(torch.square(self.k), torch.square(self.k), indexing='ij')))

    def get_adaptive_filter(self, y_f) -> torch.tensor:
        # for square domain only! augmentation for 2D of eq. 2.14 (Klein)
        spec = torch.abs(y_f) ** self.p
        kp = torch.divide(torch.trapz(torch.trapz(torch.abs(self.k * spec))), torch.trapz(torch.trapz(spec)))

        k_hat = torch.abs(torch.divide(self.grid[None, :], kp[:, None, None]))

        return torch.where(torch.greater_equal(k_hat, self.k_filter), 0, 1).type(torch.complex64)

    def get_static_filter(self) -> torch.tensor:
        return torch.where(torch.greater_equal(self.grid, self.k_filter), 0, 1).type(torch.complex64)
