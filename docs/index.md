# Getting started

!!! warning

    Clapeyron-JAX is currently unstable. Please pin your version until the v0.1.0 release.

## Installation

```bash
pip install clapeyron-jax
```

## Quick example
```py
import clapeyron_jax as cl
import equinox as eqx
import numpy as np

p, T = 1e5, 273.15

model = cl.PR(["carbon dioxide"])
jac = eqx.filter_jacfwd(lambda x: cl.mass_density(model, x[0], x[1]))

print(jac(np.array([p, T])))
```
