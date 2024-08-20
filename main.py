import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load Data:
data, target = fetch_california_housing(return_X_y=True)  # 20640 samples with 8 features
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

# Add bias (intercept) term to the features:
X_train_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]  
X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]  

# Compute the optimal theta using the Normal Equation:
theta_best = np.linalg.inv(X_train_b.T.dot(X_train_b)).dot(X_train_b.T).dot(y_train)
print(f"theta_best: \n{theta_best}")

# Make predictions on the test set:
y_pred = X_test_b.dot(theta_best)

# Evaluate the model:
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Plot Data:
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, c='blue', alpha=0.5, label="Predicted vs Actual")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], c='red', linewidth=3, label="Prediction Line")
plt.title('Actual vs. Predicted House Values')
plt.xlabel('Actual House Value')
plt.ylabel('Predicted House Value')
plt.legend()
plt.grid(True)
plt.show()
