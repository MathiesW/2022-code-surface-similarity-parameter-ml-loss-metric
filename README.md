# Surface Similarity Parameter machine learning loss metric for oscillatory spatio-temporal data
[![DOI](https://zenodo.org/badge/653051819.svg)](https://zenodo.org/badge/latestdoi/653051819)
![Tests](https://github.com/MathiesW/2022-code-surface-similarity-parameter-ml-loss-metric/actions/workflows/release.yaml/badge.svg)
![PyPI](https://img.shields.io/pypi/v/surface-similarity-parameter)
![License](https://img.shields.io/github/license/MathiesW/2022-code-surface-similarity-parameter-ml-loss-metric)

## General
This repository contains the code for the Surface Similarity Parameter (SSP) loss metric proposed in

Wedler, M., Stender, M., Klein, M., Ehlers, S. and Hoffmann, N., 2022,  "Surface Similarity Parameter: A new machine learning loss metric for oscillatory spatio-temporal data". Neural Networks 156, 123--134. DOI: https://doi.org/10.1016/j.neunet.2022.09.023

The SSP for continuous signals is a normalized error metric 

$\mathrm{SSP}(\mathbf{y},\hat{\mathbf{y}})=\frac{\sqrt{\int|F_{\mathbf{y}}(\mathbf{k}) - F_{\hat{\mathbf{y}}}(\mathbf{k})|^2d\mathbf{k}}}{\sqrt{\int|F_{\mathbf{y}}(\mathbf{k})|^2d\mathbf{k}} + \sqrt{\int|F_{\hat{\mathbf{y}}}(\mathbf{k})|^2d\mathbf{k}}}\in[0,1]$

originally introduced by Perlin & Bustamante (https://link.springer.com/article/10.1007/s10665-016-9849-7).
The SSP calculates the deviation between the signals $y$ and $\hat{y}$ in the complex Fourier domain, such that deviations in magnitude and phase are inherently penalized in a single metric.
Being a normalized error, the SSP is defined in the range [0, 1], where
- $\mathrm{SSP}=0$ indicates perfect agreement, and
- $\mathrm{SSP}=1$ indicates perfect *disagreement* among the signals.

Perfect disagreement means that either $\hat{y}=-y$, or $\hat{y}=0$ while $y\neq 0$

When used as a machine learning (ML) loss function, this aspect renders the SSP beneficial for training ML models on oscillatory spatio-temporal data. For the model to reduce the loss, the prediction has to be improved in terms of magnitude and phase. It can be shown that this results in sharper gradients, which allows the optimizer to converge faster to a better local minimum.

This package contains
- a very basic [numpy](https://numpy.org/) implementation, and
- a more complex [Keras3](https://keras.io/) SSP loss function.

### Optional lowpass filter in the loss function
The Keras3 loss function offers two different optional lowpass filters, a *static* and an *adaptive* one.
The lowpass filter is applied to the ground truth only within the loss function.
This way, the ML model is forced to suppress the high-frequency range in order to reduce the loss,
and effectively learns a lowpass filter behavior.
This is beneficial when
- the relevant dynamics are within a certain frequency band, or
- the model is trained on noisy (measurement) data.

## Installation
The package is hosted on [PyPI.org](https://pypi.org/project/surface-similarity-parameter/) and can be installed via pip
```
$ pip install surface-similarity-parameter
```
Optionally, the most recent version of [Keras3](https://keras.io/) can be automatically installed alongside using the option `[keras]`
```
$ pip install surface-similarity-parameter[keras]
```
Note that the Keras backend
- [Tensorflow](https://www.tensorflow.org/), or
- [JAX](https://docs.jax.dev/en/latest/)

has to be manually installed. **At the moment, there is no implementation for the `'torch'` backend**.


## Usage examples
Please note that more elaborate examples for each class or method can be found in the docstring of the respective class or method.

### Numpy example (metric only)
Let's start with a very basic call of the numpy implementation on two random arrays.
```
>>> from ssp.numpy import ssp
>>> import numpy as np
>>> np.random.seed(0)  # for deterministic results
>>> y1 = np.random.random((2, 32))
>>> y2 = np.random.random((2, 32))
>>> ssp(y1, y2)  # some value between 0.0 and 1.0
np.float64(0.35119514129237195)
>>> ssp(y1, y1)  # should be 0.0
np.float64(0.0)
>>> ssp(y1, -y1)  # should be 1.0
np.float64(1.0)
>>> ssp(y1, np.zeros_like(y1))  # should be 1.0
np.float64(1.0)
```

With the option `batched=True`, the SSP operates batch-wise and returns a result for each signal in a batch:
```
>>> from ssp.numpy import ssp
>>> import numpy as np
>>> np.random.seed(0)  # for deterministic results
>>> y1 = np.random.random((2, 32))
>>> y2 = np.random.random((2, 32))
>>> ssp(y1, y2, batched=True)
array([0.34864963, 0.35827101])
>>> ssp(y1, y1, batched=True)
array([0., 0.])
>>> ssp(y1, -y1, batched=True)
array([1., 1.])
>>> ssp(y1, np.zeros_like(y1), batched=True)
array([1., 1.])
```

### Keras example (SSP as an ML loss function)
Again, let's start with a very basic usage of the SSP as a Keras loss function. 
In the following example, a Sequential model is defined, build, compiled, and trained with the SSP as the loss function.
```
>>> from ssp.keras import SSP1D
>>> from keras import ops, Sequential, layers
>>> from math import pi
>>> t = ops.arange(0, 2 * pi, 2 * pi / 512)
>>> y = ops.expand_dims(ops.sin(t), axis=0)  # shape (1, 512)
>>> x = ops.ones((1, 32))  # some input data with shape (1, 32)
>>> model = Sequential([layers.Dense(64), layers.Dense(512)])
>>> model.build(input_shape=x.shape)
>>> model.compile(optimizer="adam", loss=SSP1D(lowpass="static", f=f, f_filter=2.0))
>>> model.fit(x=x, y=y, epochs=1)
```

The optional lowpass filter can help the model to focus on the relevant frequency range of the data, or omit high-frequency noise (like, e.g., in raw measurement data).
In the following example, a `static` lowpass filter with a threshold at `f_filter=2.0` is used.
When using a lowpass filter, the SSP loss function requires the FFT frequencies `f`,
which can be generated using the `ssp.keras.ops.fftshift` method, which strongly aligns with [numpy.fft.fftshift](https://numpy.org/doc/stable/reference/generated/numpy.fft.fftshift.html), but offers the optional argument `rad` to return the FFT frequencies in terms of rad instead of Hz (multiplication by $2\pi$).
```
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
```

## Citation for original paper
``` 
@article{10.1016/j.neunet.2022.09.023,
	title = {Surface similarity parameter: {A} new machine learning loss metric for oscillatory spatio-temporal data},
	volume = {156},
	issn = {0893-6080},
	shorttitle = {Surface similarity parameter},
	url = {https://www.sciencedirect.com/science/article/pii/S0893608022003732},
	doi = {10.1016/j.neunet.2022.09.023},
	abstract = {Supervised machine learning approaches require the formulation of a loss functional to be minimized in the training phase. Sequential data are ubiquitous across many fields of research, and are often treated with Euclidean distance-based loss functions that were designed for tabular data. For smooth oscillatory data, those conventional approaches lack the ability to penalize amplitude, frequency and phase prediction errors at the same time, and tend to be biased towards amplitude errors. We introduce the surface similarity parameter (SSP) as a novel loss function that is especially useful for training machine learning models on smooth oscillatory sequences. Our extensive experiments on chaotic spatio-temporal dynamical systems indicate that the SSP is beneficial for shaping gradients, thereby accelerating the training process, reducing the final prediction error, increasing weight initialization robustness, and implementing a stronger regularization effect compared to using classical loss functions. The results indicate the potential of the novel loss metric particularly for highly complex and chaotic data, such as data stemming from the nonlinear two-dimensional Kuramoto–Sivashinsky equation and the linear propagation of dispersive surface gravity waves in fluids.},
	language = {en},
	urldate = {2022-12-22},
	journal = {Neural Networks},
	author = {Wedler, Mathies and Stender, Merten and Klein, Marco and Ehlers, Svenja and Hoffmann, Norbert},
	month = dec,
	year = {2022},
	keywords = {Deep learning, Error metric, Loss function, Nonlinear dynamics, Similarity, Spatio-temporal dynamics},
	pages = {123--134},
}
 ```
