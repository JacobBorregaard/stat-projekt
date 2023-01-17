import numpy as np
from scipy.stats import binom

p_val = 2*binom.cdf(7, 15, 0.5)

print(p_val)