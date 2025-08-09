"""
Random Forest Models for Tabular Data Classification
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
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

    def plot_roc_curve(self, X_test: np.ndarray, y_test: np.ndarray) -> None:
        """
        Plot ROC curve for the classifier
        """
        if self.model is None:
            raise ValueError("Model must be trained before plotting ROC curve")
        
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Random Forest - ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, X_test: np.ndarray, y_test: np.ndarray) -> None:
        """
        Plot confusion matrix for the classifier using only matplotlib
        """
        if self.model is None:
            raise ValueError("Model must be trained before plotting confusion matrix")
        
        y_pred = self.model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        
        im = plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar(im)
        
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        verticalalignment="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=14, fontweight='bold')
        
        plt.title('Random Forest - Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)

        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['Negative', 'Positive'])
        plt.yticks(tick_marks, ['Negative', 'Positive'])
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else 0
        recall = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        plt.figtext(0.02, 0.02, 
                f'Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1-Score: {f1:.3f}', 
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        plt.tight_layout()
        plt.show()