import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

file_path = r'D:\Datasets\kc_house_data.csv'
# Load the dataset
df = pd.read_csv(file_path)

# Preview the dataset
print(df.head())

# Check for missing values
print(df.isna().sum())

# Drop the 'id' column since it's not relevant for the analysis
df.drop('id', axis=1, inplace=True)

# Convert 'date' column to datetime and extract year and month
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

# Drop the original 'date' column
df.drop('date', axis=1, inplace=True)

# Check the distribution of the target variable
sns.histplot(df['price'])
plt.show()
# Compute the correlation matrix
corr = df.corr()

# Visualize the correlation matrix as a heatmap
sns.heatmap(corr, annot=True)
plt.show()

# Split the data into training and testing sets
X = df.drop('price', axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data using standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train and evaluate multiple regression models
models = {'Linear Regression': LinearRegression(),
          'Ridge Regression': Ridge(alpha=1.0, max_iter=100000),
          'Lasso Regression': Lasso(alpha=1.0, max_iter=100000),
          'Decision Tree Regression': DecisionTreeRegressor(random_state=42),
          'Random Forest Regression': RandomForestRegressor(random_state=42),
          'Gradient Boosting Regression': GradientBoostingRegressor(random_state=42)}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print(f'{name}: RMSE = {rmse:.2f}, R^2 = {r2:.2f}')

# Tune the hyperparameters of the best model using GridSearchCV
param_grid = {'n_estimators': [100, 200, 300],
              'max_depth': [None, 5, 10, 20],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 2, 4]}

rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Evaluate the best model on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(f'Best Model (Random Forest Regression): RMSE = {rmse:.2f}, R^2 = {r2:.2f}')
