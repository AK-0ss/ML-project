# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. Sample Dataset
data = {
    'SquareFootage': [1400, 1600, 1700, 1875, 1100, 1550, 2350, 2450, 1425, 1700],
    'Bedrooms': [3, 3, 3, 4, 2, 3, 4, 4, 3, 3],
    'Bathrooms': [2, 2, 2, 3, 1, 2, 3, 3, 2, 2],
    'Price': [245000, 312000, 279000, 308000, 199000, 219000, 405000, 324000, 319000, 255000]
}

# Create DataFrame
df = pd.DataFrame(data)

# Combine Bedrooms and Bathrooms into a single feature: 'Rooms'
df['Rooms'] = df['Bedrooms'] + df['Bathrooms']

# Features (SquareFootage, Rooms) and Target (Price)
X = df[['SquareFootage', 'Rooms']]
y = df['Price']

# Visualize the data relationships (Optional)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(df['SquareFootage'], y, color='blue')
plt.xlabel('Square Footage')
plt.ylabel('Price')
plt.title('Square Footage vs Price')

plt.subplot(1, 2, 2)
plt.scatter(df['Rooms'], y, color='green')
plt.xlabel('Rooms')
plt.ylabel('Price')
plt.title('Rooms vs Price')

plt.tight_layout()
plt.show()

# 2. Normalize the Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Split the Dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. Train the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Model Evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# 6. Prediction for a New House
# Input: SquareFootage=1800, Bedrooms=3, Bathrooms=2
new_data = pd.DataFrame([[1800, 3 + 2]], columns=['SquareFootage', 'Rooms'])  # Combine Bedrooms and Bathrooms
new_data_scaled = scaler.transform(new_data)  # Scale the input

predicted_price = model.predict(new_data_scaled)
print("Predicted House Price:", predicted_price[0])
