# ==========================================
# House Price Prediction
# Author: Syed Muhammad Sadiq
# ==========================================

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Create Simple Dataset
data = {
    "Size_sqft": [1000, 1500, 2000, 2500, 3000],
    "Bedrooms": [2, 3, 3, 4, 4],
    "Price": [200000, 300000, 400000, 500000, 600000]
}

df = pd.DataFrame(data)

X = df[["Size_sqft", "Bedrooms"]]
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Predicted Prices:", predictions)
print("Actual Prices:", list(y_test))
print("MAE:", mean_absolute_error(y_test, predictions))