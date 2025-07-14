# Surface Similarity Parameter machine learning loss metric for oscillatory spatio-temporal data
[![DOI](https://zenodo.org/badge/653051819.svg)](https://zenodo.org/badge/latestdoi/653051819)

## General
This repository contains the code for the Surface Similarity Parameter (SSP) loss metric proposed in

Wedler, M., Stender, M., Klein, M., Ehlers, S. and Hoffmann, N., 2022,  "Surface Similarity Parameter: A new machine learning loss metric for oscillatory spatio-temporal data". Neural Networks 156, 123--134. DOI: https://doi.org/10.1016/j.neunet.2022.09.023

The SSP loss function is a normalized error metric defined by $J_{\mathrm{SSP}}(\mathbf{y},\hat{\mathbf{y}})=\frac{\sqrt{\int|F_{\mathbf{y}}(\mathbf{k}) - F_{\hat{\mathbf{y}}}(\mathbf{k})|^2d\mathbf{k}}}{\sqrt{\int|F_{\mathbf{y}}(\mathbf{k})|^2d\mathbf{k}} + \sqrt{\int|F_{\hat{\mathbf{y}}}(\mathbf{k})|^2d\mathbf{k}}}\in[0,1]$. Since the error is derived in the complex Fourier Space, the SSP loss function penalizes deviations in amplitude, frequency and phase at once.


## Usage
The repository contains code 
- to use the SSP as a [Keras3](https://keras.io/) loss function, and
- a simpler [numpy](https://numpy.org/) code to use the SSP as a metric.

The Keras implementation subclasses '''keras.losses.Loss''' and can be used as a loss function in '''model.compile'''.
This implementation allows the user to define a lowpass filter, which forces the model to suppress the high frequency range either adaptively ('''lowpass="adaptive"''') or with a fixed frequency threshold ('''lowpass="static"''').
Both options require the definition of a frequency vector '''f''' that matches the shape of the data.

The numpy implementation uses pure numpy code and provides only basic functionality (no filtering etc.).

### Keras example
Simple example without any lowpass filter
```
from ssp.keras import SSP1D  # or SSP2D for 2D data
from keras import Sequential, layers

model = Sequential([layers.Dense(32), layers.Dense(32)])
model.compile(
    loss=SSP1D(),
    ...
)

model.fit(...)
```

Example with static lowpass filter
```
from ssp.keras import SSP1D  # or SSP2D for 2D data
from ssp.keras.ops import fftfreq  # method to generate a frequency vector
from keras import Sequential, layers

f = fftfreq(n=32, d=0.1, rad=False)  # provided the data is one-dimensional time series of length 32 with a temporal sampling of 0.1s

model = Sequential([layers.Dense(32), layers.Dense(32)])
model.compile(
    loss=SSP1D(
        lowpass="static",
        f=f,
        f_filter=2.0,  # high frequency range of ground truth greater than f=f_filter is set to 0.0
    ),
    ...
)
```

### numpy example
```
from ssp.numpy import ssp
import numpy as np

x1 = np.random.random((2, 32, 32))
x2 = np.random.random((2, 32, 32))

e = ssp(x1, x2)  # result in condensed to a single float value
e_batched = ssp(x1, x2, batched=True)  # one float value for each item in the batch (first dimension of the data, here, batchsize == 2)
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
