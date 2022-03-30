# Normalization (RidgeClassifierCV)

## "Old" Method (sklearn v0.24)

Feature normalization was previously (sklearn v0.24) achieved using the now-deprecated `normalize` parameter of `RidgeClassifierCV`, such that ["the regressors X [are] normalized... by subtracting the mean and dividing by the l2-norm"](https://scikit-learn.org/0.24/modules/generated/sklearn.linear_model.RidgeClassifierCV.html?highlight=ridgeclassifiercv#sklearn.linear_model.RidgeClassifierCV).

```python
classifier = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10), normalize = True)
classifier.fit(X_training_transform, Y_training)
```

## "New" Method (sklearn v1.0+)

In order to replicate the behaviour of the `normalize` parameter, feature normalization should now (sklearn v1.0+) be performed "manually" by subtracting the (per feature/column) **mean**, and dividing by the (per feature/column) **l2 norm**.

Note:
* The mean and l2 norm should be estimated using the training data.
* The l2 norm must be computed *after* subtracting the mean.
* The current (v1.0.2) sklearn documentation and deprecation warnings are ***wrong***.  You should ***not*** use `StandardScaler` in order to reproduce the behaviour of the `normalize` parameter, as `StandardScaler` uses the standard deviation (not the l2 norm), and will produce different results.  (Using `Normalizer` is also inappropriate, as it acts on rows/examples, not columns/features.)

```python
_mean = X_training_transform.mean(0)
_norm = (X_training_transform - _mean).norm(dim = 0) + 1e-8

X_training_transform = (X_training_transform - _mean) / _norm
X_test_transform = (X_test_transform - _mean) / _norm

classifier = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10))
classifier.fit(X_training_transform, Y_training)
```
