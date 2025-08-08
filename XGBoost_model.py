

import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score, roc_auc_score, mean_squared_error,
    roc_curve, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import KFold

# Optional SHAP explanation (will plot if available)
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

def _plot_feature_importance(model, feature_names=None, top_n=10):
    """Bar plot of model.feature_importances_ using matplotlib (no seaborn)."""
    importance = model.feature_importances_
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(len(importance))]
    df = pd.DataFrame({"feature": feature_names, "importance": importance})
    df = df.sort_values("importance", ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    plt.barh(df["feature"][::-1], df["importance"][::-1])
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.show()

def train_xgboost_classifier(X, y, feature_names=None, n_splits=5, plot=True, save_model=True):
    """
    Train an XGBoost classifier with K-Fold CV, plots, SHAP (if available), and model saving.

    Comments:
    - Confusion matrix is plotted for each fold to inspect errors.
    - ROC curve is plotted across folds.
    - Feature importance is plotted from the last fold's model.
    - SHAP summary plot is included if the 'shap' package is available.
    - Model is saved via joblib after training.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    acc_scores, auc_scores = [], []
    all_fprs, all_tprs, all_aucs = [], [], []
    last_model = None

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

        # Log parameters so it's visible what we're using
        print(f"[Fold {i+1}] XGBClassifier params:", model.get_params())  # <-- parameter logging

        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=10,  # <-- early stopping
            verbose=False
        )

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc_scores.append(accuracy_score(y_test, y_pred))
        auc_score = roc_auc_score(y_test, y_proba)
        auc_scores.append(auc_score)

        if plot:
            # Confusion matrix per fold
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot()
            plt.title(f"Confusion Matrix - Fold {i+1}")
            plt.tight_layout()
            plt.show()

            # ROC data per fold (plotted after CV loop)
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            all_fprs.append(fpr)
            all_tprs.append(tpr)
            all_aucs.append(auc_score)

        last_model = model  # keep reference to the last trained model

    print(f"XGBoost Classification (KFold={n_splits}):")
    print(f"  Accuracy: {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}")
    print(f"  AUC:      {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")

    if plot and all_fprs:
        plt.figure(figsize=(8, 6))
        for i in range(n_splits):
            plt.plot(all_fprs[i], all_tprs[i], label=f"Fold {i+1} AUC={all_aucs[i]:.2f}")
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves for XGBoost Classifier")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Feature importance from the last fold's model
        _plot_feature_importance(last_model, feature_names)

        # SHAP summary plot (optional)
        if SHAP_AVAILABLE:
            print("Generating SHAP summary plot...")  # <-- SHAP info
            # For speed, sample if very large
            X_sample = X if X.shape[0] <= 5000 else X[np.random.choice(X.shape[0], 5000, replace=False)]
            explainer = shap.Explainer(last_model, X_sample)
            shap_values = explainer(X_sample)
            shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=True)
        else:
            print("SHAP not available, skipping explanation plot.  # <-- Install 'shap' to enable")

    if save_model and last_model is not None:
        joblib.dump(last_model, "xgboost_classifier_model.joblib")
        print("Saved: xgboost_classifier_model.joblib  # <-- Load later with joblib.load(...)")

    return last_model

def train_xgboost_regressor(X, y, feature_names=None, n_splits=5, plot=True, save_model=True):
    """
    Train an XGBoost regressor with K-Fold CV, plots, and model saving.

    Comments:
    - Predicted vs True plot shows how well the model fits across folds.
    - Feature importance is plotted from the last fold's model.
    - Model is saved via joblib after training.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmse_scores, preds_all, trues_all = [], [], []
    last_model = None

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

        print(f"[Fold {i+1}] XGBRegressor params:", model.get_params())  # <-- parameter logging

        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=10,  # <-- early stopping
            verbose=False
        )

        y_pred = model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        rmse_scores.append(rmse)

        preds_all.extend(y_pred)
        trues_all.extend(y_test)
        last_model = model

    print(f"XGBoost Regression (KFold={n_splits}):")
    print(f"  RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")

    if plot and len(preds_all) > 0:
        plt.figure(figsize=(8, 6))
        plt.scatter(trues_all, preds_all, alpha=0.3)
        lo = min(min(trues_all), min(preds_all))
        hi = max(max(trues_all), max(preds_all))
        plt.plot([lo, hi], [lo, hi])
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        plt.title("Predicted vs True Values (XGBoost Regression)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        _plot_feature_importance(last_model, feature_names)

    if save_model and last_model is not None:
        joblib.dump(last_model, "xgboost_regressor_model.joblib")
        print("Saved: xgboost_regressor_model.joblib  # <-- Load later with joblib.load(...)")

    return last_model

