import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Historical financial data
data = pd.DataFrame({
    "Stocks": [20, 25, 30, 35, 40, 45, 50, 55, 60, 65],
    "Bonds": [60, 55, 50, 45, 40, 35, 30, 25, 20, 15],
    "Gold": [20, 20, 20, 20, 20, 20, 20, 20, 20, 20],
    "Return": [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
})

# Features and target
X = data[["Stocks", "Bonds", "Gold"]]
y = data["Return"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Prediction
predicted = model.predict(X_test)

print("Predicted Portfolio Returns:")
print(np.round(predicted, 2))

# Model evaluation
mse = mean_squared_error(y_test, predicted)

print("Mean Squared Error:", round(mse, 2))

# Compare two portfolio allocations
portfolio1 = [[60, 30, 10]]
portfolio2 = [[30, 50, 20]]

prediction1 = model.predict(portfolio1)[0]
prediction2 = model.predict(portfolio2)[0]

print("Portfolio 1 Predicted Return:",
      round(prediction1, 2))

print("Portfolio 2 Predicted Return:",
      round(prediction2, 2))

if prediction1 > prediction2:
    print("Best Portfolio: Portfolio 1")
else:
    print("Best Portfolio: Portfolio 2")
