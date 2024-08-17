import numpy as np
import pandas as pd

x = np.linspace(1, 100, 100)
y = np.linspace (101, 200, 100)
delta = np.random.uniform(-10, 10, x.size)
z = 2.3 * x - 0.4 * y + delta

data = pd.DataFrame(
    {
        'x': pd.Series(x),
        'y': pd.Series(y),
        'z': pd.Series(z)
    }
)

data.to_csv('data.csv', index=False)