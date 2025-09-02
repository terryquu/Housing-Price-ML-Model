import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

csv_path = "Housing-Price-ML-Model/kaggle_housing.csv"
df = pd.read_csv(csv_path)

price_per_room = np.zeros(5000, dtype=float)
price_series = df['Price']
rooms_series = df['Avg. Area Number of Rooms']

df['Price Per Room'] = df['Price'] / df['Avg. Area Number of Rooms']

X = df[['Price Per Room', 'Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print("MSE: ", mean_squared_error(y_test, y_pred))
print("R2: ", r2_score(y_test, y_pred))

importances = rf.feature_importances_
features = X.columns

plt.bar(features, importances)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Feature Importance In Random Forest")
plt.show()