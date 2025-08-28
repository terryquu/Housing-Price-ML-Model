import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

csv_path = "Housing-Price-ML-Model/kaggle_housing.csv"
df = pd.read_csv(csv_path)

price_per_room = np.zeros(5000, dtype=float)
price_series = df['Price']
rooms_series = df['Avg. Area Number of Rooms']

df['Price Per Room'] = data['Price'] / data['Avg. Area Number of Rooms']

X = df[['Price Per Room', 'Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 'Price']]
