import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, Binarizer
from sklearn.metrics import mean_squared_error, r2_score

california = fetch_california_housing(as_frame=True)  

# Use `as_frame=True` for easier manipulation
data = california.frame
print(data.head())

X = data.drop(columns=['MedHouseVal']) # Features
y = data['MedHouseVal'] # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler_mean = StandardScaler(with_mean=True, with_std=False)
X_train_mean_removed = scaler_mean.fit_transform(X_train)
X_test_mean_removed = scaler_mean.transform(X_test)

scaler_minmax = MinMaxScaler()
X_train_scaled = scaler_minmax.fit_transform(X_train)
x_test_scaled = scaler_minmax.transform(X_test)

normalizer = Normalizer()
X_train_normalized = normalizer.fit_transform(X_train)
X_test_normalized = normalizer.transform(X_test)

binarizer = Binarizer(threshold=0.5)
X_train_binarized = binarizer.fit_transform(X_train)
X_text_binarizer = binarizer.transform(X_test)

model = LinearRegression()
model.fit(X_train_mean_removed, y_train)

y_pred = model.predict(X_test_mean_removed)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("In California housing linear progression performance: ")
print(f"Mean squared error: {mse}")
print(f"R-Squared: {r2}")