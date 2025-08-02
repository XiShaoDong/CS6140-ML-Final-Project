
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, roc_auc_score, mean_squared_error,
    roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import KFold, GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_feature_importance(model, feature_names=None, top_n=10):
    importance = model.feature_importances_
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(len(importance))]
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    importance_df = importance_df.sort_values(by='Importance', ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.show()

def train_xgboost_classifier(X, y, feature_names=None, n_splits=5, plot=True):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    acc_scores, auc_scores = [], []
    all_fprs, all_tprs, all_aucs = [], [], []

    for i, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=10,
            verbose=False
        )

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc_scores.append(accuracy_score(y_test, y_pred))
        auc = roc_auc_score(y_test, y_proba)
        auc_scores.append(auc)

        if plot:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            all_fprs.append(fpr)
            all_tprs.append(tpr)
            all_aucs.append(auc)

    print(f"XGBoost Classification (KFold={n_splits}):")
    print(f"  Accuracy: {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}")
    print(f"  AUC:      {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")

    if plot:
        plt.figure(figsize=(8, 6))
        for i in range(n_splits):
            plt.plot(all_fprs[i], all_tprs[i], label=f"Fold {i+1} AUC={all_aucs[i]:.2f}")
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves for XGBoost Classifier")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Also plot feature importance for final model
        plot_feature_importance(model, feature_names)

    return model

def train_xgboost_regressor(X, y, feature_names=None, n_splits=5, plot=True):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmse_scores, predictions, true_vals = [], [], []

    for i, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=10,
            verbose=False
        )
        y_pred = model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        rmse_scores.append(rmse)
        predictions.extend(y_pred)
        true_vals.extend(y_test)

    print(f"XGBoost Regression (KFold={n_splits}):")
    print(f"  RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")

    if plot:
        plt.figure(figsize=(8, 6))
        plt.scatter(true_vals, predictions, alpha=0.3)
        plt.plot([min(true_vals), max(true_vals)], [min(true_vals), max(true_vals)], color='red')
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        plt.title("Predicted vs True Values (Regression)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Plot feature importance
        plot_feature_importance(model, feature_names)

    return model
