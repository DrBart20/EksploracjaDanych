import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

def train_model(X_train: pd.DataFrame, 
                y_train: pd.Series, 
                model_name: str, 
                params: dict) -> Pipeline:
   
    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Random Forest Regressor":
        model = RandomForestRegressor(random_state=42, **params)
    elif model_name == "SVR":
        model = SVR(**params)
    else:
        print(f"Nieznana nazwa modelu '{model_name}'.")
        model = LinearRegression()

    pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    
    pipeline.fit(X_train, y_train)
    
    return pipeline

def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        "R²": r2,
        "RMSE": rmse
    }
    
    return metrics

def make_prediction(model: Pipeline, input_data: pd.DataFrame) -> np.ndarray:
    prediction = model.predict(input_data)
    
    return prediction
