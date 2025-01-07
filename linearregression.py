import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from preproc import load_data, preprocessing
from sklearn.feature_selection import RFE

folder_path = 'C:/AIML_ass2'

df = load_data(folder_path)
df_proc = preprocessing(df)

correlation_matrix = df_proc.corr()

print("\nTop Correlations with Precipitation (prec):")
print(correlation_matrix['prec'].sort_values(ascending=False))

features = [
    '10m_wind_u', 'lai_high_veg', '2m_dp_temp_mean',
    '2m_dp_temp_min', '2m_dp_temp_max', '2m_temp_min', 'surf_net_therm_rad_mean',
    'surf_net_therm_rad_max', 'surf_net_solar_rad_max', 'surf_press', 'DOY_cos', 'MM_cos',
    'DOY_sin', 'MM_sin', '10m_wind_v'
]

X = df_proc[features]
y = df_proc['prec']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

# Model training and evaluation
model = LinearRegression()

# Stepwise selection
selector = RFE(model, n_features_to_select=15)
selector = selector.fit(X_train, y_train)

selected_features = X.columns[selector.support_]
print("Selected Features:", selected_features)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")


plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red')  # Ideal line
plt.xlabel('Actual Precipitation')
plt.ylabel('Predicted Precipitation')
plt.title('Predicted vs Actual Precipitation')
plt.show()
