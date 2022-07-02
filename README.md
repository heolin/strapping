# Strapping [![Build Status](https://travis-ci.com/heolin123/strapping.svg?branch=master)](https://travis-ci.com/heolin123/strapping)
Strapping is a library containing a fast implementation of bootstrapping sampling algorithm.
Along the sampling algorithms you will find a set of helper functions used to compute basic statistics
useful in bootstrapping-based analysis.

Library supports:
- single variable sampling
- multi-column variable sampling
- A/B test difference sampling

## Installing
Strapping can be installed via pip from [PyPI](https://pypi.org/project/strapping/).

```bash
pip install strapping
```

## Testing
Tu run tests for the package use `tox`:
```bash
tox
```

# Example
## Sample single variable
In this example we will use a bootstrapping algorithm to sample a distribution of mean and std. deviation of the given dataset.

### Sample means using bootstrapping
Import `bootstrap` and `stats` module.
- `bootstrap` contains bootstrapping algorithms,
- `stats` contains helpers for computing basic statistics (e.g. confidence intervals).

```python
from strapping import bootstrap, stats
```

Generate sample data using normal distribution:
```python
X = np.random.normal(0, 1, size=100).reshape(-1, 1)
```

Sample a vector containing possible means for given dataset:

```python
mu_sampled = bootstrap.sample(X, iterations=1000, aggrfunc=np.mean)
std_sampled = bootstrap.sample(X, iterations=1000, aggrfunc=np.std)
```

We can check output values:
```python
>>> np.mean(mu_sampled), np.mean(std_sampled)
(-0.028259915654785906, 1.0099170040429664)
```

### Compute confidence intervals
Now we will compute confidence intervals based on sampled values. This works for both single values and multi-column variables.
By default, confidence interval will three values: (5th quantile, mean, 95th quantile).

```python
q05, mean, q95 = stats.confidence_intervals(mu_sampled)
```

We can check output values:
```python
>>> q05
array([-0.15844911])

>>> mean
array([-0.01509199])

>>> q95
array([0.12659994])
```

## Sample multi-column variables
In this example we will test using bootstrapping for data containing multiple columns.

Generate data containing multiple columns:
```python
X = np.array([
    np.random.normal(0, 1, size=100),
    np.random.normal(10, 5, size=100),
    np.random.normal(-20, 5, size=100),
]).T
```

Import `bootstrap` module:
```python
from strapping import bootstrap 
```

Sample mean for given dataset:
```python
mu_sampled = bootstrap.sample(X, iterations=1000, aggrfunc=np.mean)
```

We can check output values:
```python
>>> mu_sampled.mean(axis=0)
array([ -0.06588892,   9.97571153, -19.187514  ])
```

## A/B test difference between two variables
In this example we will test using bootstrapping to sample a difference between two given datasets.
Then, we will use sampled values to compute percentage confidence intervals for the difference.

### Sample means using bootstrapping
Generate data containing multiple columns:
```python
X1 = np.random.normal(5, 2, size=100).reshape(-1, 1)
X2 = np.random.normal(6, 2, size=100).reshape(-1, 1)
```

Import `bootstrap` and `stats` modules:
```python
from strapping import bootstrap, stats 
```

Sample mean for given dataset:
```python
mu_sampled = bootstrap.sample_diffs(X1, X2, iterations=1000, aggrfunc=np.mean)
```

We can check output values:
```python
>>> mu_sampled.mean()
-1.2875678613575356
```

### Compute confidence intervals
Now we will compute both confidence intervals and percentage confidence intervals based on sampled values.

```python
>>> stats.confidence_intervals(mu_sampled)
(array([-1.77019123]), array([-1.28756786]), array([-0.79820009]))
```

Percentage confidence intervals are computed as a percentage difference between sampled values and the mean value 
of a provided reference (control dataset).

```python
>>> stats.percentage_confidence_intervals(mu_sampled, X1.mean())
(array([-0.36300107]), array([-0.26403278]), array([-0.16368146]))
```

## Other
### Compute Cohen's d
Using `strapping` you can easily compute bootstrapped value of Cohen's d,
which is often used for a metric of measuring the effect size.

To do so first compute the difference between two datasets:
```python
diff_sampled = bootstrap.sample_diffs(X1, X2, iterations=1000, aggrfunc=np.mean)
```

Then, compute the pooled standard deviation using a helper function and finally compute Cohen's d value:
```python
from strapping.stats import pooled_std
pstd = pooled_std(X1, X2)

cohensd = diff_sampled / pstd
```
