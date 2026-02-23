# ================================
# Land & Home Price Prediction
# Linear Regression - AIML Assessment
# ================================

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 2. Load Dataset (California Housing)
data = fetch_california_housing(as_frame=True)
df = data.frame

# 3. Data Understanding
print(df.head())
print(df.info())
print(df.describe())

# Independent & Dependent variables
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# 4. Data Preprocessing
# (No missing values in this dataset)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Model Building
model = LinearRegression()
model.fit(X_train, y_train)

print("Intercept:", model.intercept_)
print("Coefficients:")
coeff_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})
print(coeff_df)

# Predictions
y_pred = model.predict(X_test)

# 6. Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

# 7. Visualization

# Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Actual vs Predicted
plt.figure(figsize=(6,5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()

# Feature Importance (Coefficients)
plt.figure(figsize=(8,5))
sns.barplot(x='Coefficient', y='Feature', data=coeff_df)
plt.title("Feature Importance")
plt.show()

# 8. Interpretation & Business Insights
most_influential = coeff_df.iloc[coeff_df['Coefficient'].abs().argmax()]
print("Most Influential Feature:", most_influential['Feature'])

print("""
Business Insights:
1. Area-related features strongly influence house prices.
2. Higher income locations tend to have higher property values.
3. Model shows good reliability if R2 score is close to 1.
4. Accuracy can be improved using more features or advanced models.
""")