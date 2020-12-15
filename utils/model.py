import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import LinearSVR
from xgboost import XGBRegressor

def select_regressor(selection):
    regressors = {
    'Linear Regression': LinearRegression(),
    'K-Nearest Neighbors': KNeighborsRegressor(),
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'XGBoost': XGBRegressor(verbosity = 0),
    'Support Vector Machines': LinearSVR(),
    'Extra Trees': ExtraTreesRegressor(),
     }
    
    return regressors[selection]


def train_model(regressor, X_train, y_train):
    model = select_regressor(regressor)

    # Fit model
    model.fit(X_train, y_train) 
    return model

def evaluate_model(model, X_train, X_test, y_train, y_test):
    
    # Evaluate model
    train_preds = model.predict(X_train)
    train_preds = np.maximum(train_preds, 0) # Don't predict negative cases
    train_mae = mae(train_preds, y_train)

    test_preds = model.predict(X_test)
    test_preds = np.maximum(test_preds, 0) # Don't predict negative cases
    test_mae = mae(test_preds, y_test)

    return train_preds, train_mae, test_preds, test_mae

# Helpful function to compute mae
def mae(pred, true):
    return np.mean(np.abs(pred - true))
