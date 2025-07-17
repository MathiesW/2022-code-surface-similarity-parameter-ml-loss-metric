Surface Similarity Parameter
============================

The Surface Similarity Parameter

.. math::
   \mathrm{SSP} = \frac{\sqrt{\int|Y - \hat{Y}|^2 df}}{\sqrt{\int|Y|^2 df} + \sqrt{\int|\hat{Y}|^2 df}}\in[0, 1]
   :label: continuous

with :math:`Y` the Fourier transform of :math:`y`, is a normalized error metric originally introduced by `Perlin and Bustamante (2016) <https://doi.org/10.1007/s10665-016-9849-7>`_.
The SSP quantifies the difference between two signals in the complex Fourier space, and thus inherently penalizes deviations in magnitude and phase in a single metric.

For discrete signals, Eq. :eq:`continuous` collapses to

.. math::
   \mathrm{SSP} = \frac{||Y-\hat{Y}||}{||Y|| + ||\hat{Y}||}\in[0, 1].

Being a normalized error, the SSP is defined in the range `[0, 1]`, where

- :math:`\mathrm{SSP}=0` indicates perfect agreement, and
- :math:`\mathrm{SSP}=1` indicates perfect *disagreement* among the signals.

Perfect disagreement means that either :math:`\hat{y}=-y`, or :math:`\hat{y}=0` while :math:`y\neq 0`.

Using the SSP as a machine learning loss function forces the model to improve the prediction in terms of magnitude and phase in order to reduce the loss  (cf. `Wedler et al. (2022) <https://doi.org/10.1016/j.neunet.2022.09.023>`_).
This sets the SSP apart from established Euclidean distance-based loss functions like the MSE and MAE, when training models on oscillatory spatio-temporal data.
The stricter error penalization of the SSP leads to a more refined and *optimizer-friendly* loss surface, where local minima are sparse but meaningful. This allows the optimizer to take more confident steps, leading to faster convergence to better local minima.

Package content
---------------

.. toctree::
   :maxdepth: 3

   ssp.keras.main
   ssp.numpy.main
