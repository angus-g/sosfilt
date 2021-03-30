# sosfilt

Second-order section filtering with multiple filters.

This library just extracts scipy's `sosfiltfilt` (and related)
method. Instead of taking a single SOS filter, with shape
`(n_sections, 6)`, it takes one filter for each core slice of the
input array. That is, the SOS filter has shape `(n_filters,
n_sections, 6)`, and the input array has shape `(..., n, ...`), where
the product of the shape of all the elided dimensions is
`n_filters`. Notably, it will not do any broadcasting along the
non-core dimensions.

The actual filtering algorithm employed within this library is exactly
the same as that within scipy. In that sense, results should be
identical. However, in order to vectorise the creation of the initial
conditions with `sosfilt_zi`, a direct calculation is employed, rather
than deferring to a linear solve. The result should be practically
identical, but it is possible that they aren't reproducible to machine
precision due to this adjustment.

## Installation

Install directly from the git repository, install from pypi with `pip
install sosfilt`, or install from conda with `conda install -c angus-g
sosfilt`.

## Example

```python
import sosfilt
import numpy as np

x = np.random.randn(5, 20, 5)
# 25 random lowpass filters
f = sosfilt.butter(3, np.random.rand(25))

sosfilt.sosfiltfilt(f, x, axis=1)
```
