"""
Random Forest Models for Tabular Data Classification
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from typing import Dict, List, Any

class RandomForestBinaryClassifier:
    """
    Random Forest Binary Classifier with hyperparameter tuning and evaluation
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
        
        rf = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
        
        grid_search = GridSearchCV(
            rf, param_grid, cv=cv_folds, 
            scoring='roc_auc', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        self.best_params = grid_search.best_params_
        
        return self.best_params
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
             use_tuned_params: bool = True) -> None:
        """
        Train the Random Forest classifier
        """
        if use_tuned_params and self.best_params:
            self.model = RandomForestClassifier(**self.best_params, random_state=self.random_state)
        else:
            # Default robust parameters
            self.model = RandomForestClassifier(
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
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        return metrics
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv_folds: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation
        """
        if self.model is None:
            # Use default model for cross-validation
            model = RandomForestClassifier(
                n_estimators=200, max_depth=20, random_state=self.random_state
            )
        else:
            model = self.model
        
        cv_accuracy = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy')
        cv_f1 = cross_val_score(model, X, y, cv=cv_folds, scoring='f1')
        cv_auc = cross_val_score(model, X, y, cv=cv_folds, scoring='roc_auc')
        
        self.cv_scores = {
            'accuracy_mean': cv_accuracy.mean(),
            'accuracy_std': cv_accuracy.std(),
            'f1_mean': cv_f1.mean(),
            'f1_std': cv_f1.std(),
            'auc_mean': cv_auc.mean(),
            'auc_std': cv_auc.std()
        }
        
        return self.cv_scores
    
    def plot_feature_importance(self, feature_names: List[str], top_n: int = 15) -> None:
        """
        Plot feature importance
        """
        if self.feature_importance is None:
            raise ValueError("Model must be trained to plot feature importance")
        
        # Get top N features
        indices = np.argsort(self.feature_importance)[::-1][:top_n]
        top_features = [feature_names[i] for i in indices]
        top_importance = self.feature_importance[indices]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_importance)
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importances - Random Forest Classifier')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
