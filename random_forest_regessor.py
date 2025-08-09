"""
Random Forest Models for Tabular Data  Regression
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor as SklearnRF
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from typing import Dict, List, Any



class RandomForestRegressor:
    """
    Random Forest Regressor with hyperparameter tuning and evaluation
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.feature_importance = None
        self.cv_scores = None
        
    def tune_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray, 
                           cv_folds: int = 5) -> Dict[str, Any]:
        """
        Perform grid search for hyperparameter tuning
        """
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        rf = SklearnRF(random_state=self.random_state, n_jobs=-1)
        
        grid_search = GridSearchCV(
            rf, param_grid, cv=cv_folds, 
            scoring='neg_mean_squared_error', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        self.best_params = grid_search.best_params_
        
        return self.best_params
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
             use_tuned_params: bool = True) -> None:
        """
        Train the Random Forest regressor
        """
        if use_tuned_params and self.best_params:
            self.model = SklearnRF(**self.best_params, random_state=self.random_state)
        else:
            self.model = SklearnRF(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=self.random_state,
                n_jobs=-1
            )
        
        self.model.fit(X_train, y_train)
        self.feature_importance = self.model.feature_importances_
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on test data
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred)
        }
        
        return metrics
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv_folds: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation
        """
        if self.model is None:
            model = SklearnRF(
                n_estimators=200, max_depth=20, random_state=self.random_state
            )
        else:
            model = self.model
        
        cv_mse = -cross_val_score(model, X, y, cv=cv_folds, scoring='neg_mean_squared_error')
        cv_r2 = cross_val_score(model, X, y, cv=cv_folds, scoring='r2')
        
        self.cv_scores = {
            'rmse_mean': np.sqrt(cv_mse.mean()),
            'rmse_std': np.sqrt(cv_mse.std()),
            'r2_mean': cv_r2.mean(),
            'r2_std': cv_r2.std()
        }
        
        return self.cv_scores
    
    def plot_predictions(self, X_test: np.ndarray, y_test: np.ndarray) -> None:
        """
        Plot predictions vs actual values
        """
        if self.model is None:
            raise ValueError("Model must be trained before plotting predictions")
        
        y_pred = self.model.predict(X_test)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Random Forest: Predictions vs Actual Values')
        plt.tight_layout()
        plt.show()
        
        # Residual plot
        residuals = y_test - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Random Forest: Residual Plot')
        plt.tight_layout()
        plt.show()

