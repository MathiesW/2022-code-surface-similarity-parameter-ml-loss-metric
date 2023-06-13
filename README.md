# Surface Similarity Parameter machine learning loss metric for oscillatory spatio-temporal data

##

## General
This repository contains the code for the Surface Similarity Parameter (SSP) loss metric proposed in

Wedler, M., Stender, M., Klein, M., Ehlers, S. and Hoffmann, N., 2022,  "Surface Similarity Parameter: A new machine learning loss metric for oscillatory spatio-temporal data". Neural Networks 156, 123--134. DOI: https://doi.org/10.1016/j.neunet.2022.09.023

The SSP loss function is a normalized error metric defined by $J_{\mathrm{SSP}}(\mathbf{y},\hat{\mathbf{y}})=\frac{\sqrt{\int|F_{\mathbf{y}}(\mathbf{k}) - F_{\hat{\mathbf{y}}}(\mathbf{k})|^2d\mathbf{k}}}{\sqrt{\int|F_{\mathbf{y}}(\mathbf{k})|^2d\mathbf{k}} + \sqrt{\int|F_{\hat{\mathbf{y}}}(\mathbf{k})|^2d\mathbf{k}}}\in[0,1]$. Since the error is derived in the complex Fourier Space, the SSP loss function penalizes deviations in amplitude, frequency and phase at once.


## Usage
The code is provided for TensorFlow (by subclassing ```tf.keras.losses.Loss```) and Pytorch (by subclassing ```nn.module```). All code is written using methods from the respective NN library, such that no additional packages are required to use the SSP loss function for training either a TensorFlow or a Pytorch model.

Both implementations come for one- and two-dimensional data. Besides the *vanilla SSP* which is presented and discussed in the publication, there are modified versions that apply a low pass filter to the ground truth during training. This small change enforces the model to learn a low pass filter, i.e., suppress high frequency noise in order to minimize the loss. Overall, there are
- ```SurfaceSimilarityParameter```: vanilla SSP for one-dimensional data
- ```SurfaceSimilarityParameterLowPass```: SSP with either *adaptive* or *static* low pass filter for one-dimensional data
- ```SurfaceSimilarityParameter2D```: vanilla SSP for two-dimensional data
- ```SurfaceSimilarityParameter2DLowPass```: SSP with either *adaptive* or *static* low pass filter for two-dimensional data

The two-dimensional implementations currently only support squared domains.

### Usage example for TensorFlow
The SSP implementation for TensorFlow can be imported from ```ssp/ssp_tensorflow```. The SSP formulation are written using TensorFlow methods only, such that it works as any other TensorFlow loss metric:
```
from ssp.ssp_tensorflow import SurfaceSimilarityParameter
ssp = SurfaceSimilarityParameter()
model.compile(loss=ssp, ...)
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
