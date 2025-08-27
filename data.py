import pandas as pd
from sklearn.datasets import fetch_california_housing

cali_housing = fetch_california_housing(as_frame = True)

print(cali_housing.frame.head())