[`ROCKET`](https://github.com/angus924/rocket) &middot; [`MINIROCKET`](https://github.com/angus924/minirocket) &middot; [**`HYDRA`**](https://github.com/angus924/hydra)

# HYDRA

***HYDRA: Competing convolutional kernels for fast and accurate time series classification***

[arXiv:2203.13652](https://arxiv.org/abs/2203.13652) (preprint)


> <div align="justify">We demonstrate a simple connection between dictionary methods for time series classification, which involve extracting and counting symbolic patterns in time series, and methods based on transforming input time series using convolutional kernels, namely ROCKET and its variants.  We show that by adjusting a single hyperparameter it is possible to move by degrees between models resembling dictionary methods and models resembling ROCKET.  We present HYDRA, a simple, fast, and accurate dictionary method for time series classification using competing convolutional kernels, combining key aspects of both ROCKET and conventional dictionary methods.  HYDRA is faster and more accurate than the most accurate existing dictionary methods, and can be combined with ROCKET and its variants to further improve the accuracy of these methods.</div>

Please cite as:

```bibtex
@article{dempster_etal_2022,
  author  = {Dempster, Angus and Schmidt, Daniel F and Webb, Geoffrey I},
  title   = {{HYDRA}: Competing convolutional kernels for fast and accurate time series classification},
  year    = {2022},
  journal = {arXiv:2203.13652}
}
```

## Results

#### UCR Archive (112 Datasets, 30 Resamples)

* [Hydra](./results/results_ucr112_hydra.csv)
* [Rocket+Hydra, MiniRocket+Hydra, MultiRocket+Hydra](./results/results_ucr112_variants.csv)

#### Sensitivity Analysis (Additional Results)

* [Accuracy vs *k* & *g*](./results/accuracy_vs_k_and_g.pdf) (pdf)

## Requirements

* Python
* PyTorch
* NumPy
* scikit-learn (or similar)

## Code

### [`hydra.py`](./code/hydra.py)
### [`hydra_multivariate.py`](./code/hydra_multivariate.py)&#8224;
### [`softmax.py`](./code/softmax.py)\*

&#8224; *experimental*  
\* *Hydra + SGD for larger datasets (i.e., more than approx. 10,000 training examples)*

## Examples

```python
from hydra import Hydra
from sklearn.linear_model import RidgeClassifierCV

[...] # load data (torch.FloatTensor, shape = (num_examples, 1, length))

transform = Hydra(X_training.shape[-1])

X_training_transform = transform(X_training)
X_test_transform = transform(X_test)

classifier = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10), normalize = True) # see note
classifier.fit(X_training_transform, Y_training)

predictions = classifier.predict(X_test_transform)
```

<details><summary><b>Note re Normalization</b> <i>(click to open)</i></summary>

To reproduce the behaviour of the (now deprecated) `normalize` parameter of `RidgeClassifierCV`, subtract the (per feature/column) mean and divide by the (per column/feature) l2 norm.

```python
_mean = X_training_transform.mean(0)
_norm = (X_training_transform - _mean).norm(dim = 0) + 1e-8

X_training_transform = (X_training_transform - _mean) / _norm
X_test_transform = (X_test_transform - _mean) / _norm

classifier = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10))
classifier.fit(X_training_transform, Y_training)
```
</details>

## Acknowledgements

We thank Professor Eamonn Keogh and all the people who have contributed to the UCR time series classification archive.  Figures in our paper showing mean ranks were produced using code from [Ismail Fawaz et al. (2019)](https://github.com/hfawaz/cd-diagram).

<div align="center">:dragon_face:</div>
