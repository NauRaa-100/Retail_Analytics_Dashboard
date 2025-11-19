
from rfm import rfm
import numpy as np
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error 
import joblib

# Features & Target
X = rfm[['Recency','Frequency','Monetary']]
y = rfm['Monetary']  

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline
clv_pipeline = Pipeline([
    ('scaler', RobustScaler()),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train
clv_pipeline.fit(X_train, y_train)

# Predict & evaluate
y_pred = clv_pipeline.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("CLV Prediction RMSE:", rmse)

# Save model
joblib.dump(clv_pipeline, "clv_pipeline.pkl")
print("CLV pipeline saved!")