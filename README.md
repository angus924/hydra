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

## Examples

## Acknowledgements

We thank Professor Eamonn Keogh and all the people who have contributed to the UCR time series classification archive.  Figures in our paper showing mean ranks were produced using code from [Ismail Fawaz et al. (2019)](https://github.com/hfawaz/cd-diagram).

<div align="center">:dragon_face:</div>
