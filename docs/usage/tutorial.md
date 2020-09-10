## Tutorial

We create a simple problem, with trajectories in 2 dimensions, a few constraint functions and an explicit cost function.

First, we need to import Pyrotor and Numpy to define our problem

```python
import numpy as np
import pyrotor
```

Then we can define our quadratic cost function, let's say:

```python
# Quadratic part
q = np.array([[1,0],
              [0,1]])
# Linear part
w = np.array([0,0])

# Constant part
c = 2.87

quadratic_model = [c, w, q]
```

And so on...
