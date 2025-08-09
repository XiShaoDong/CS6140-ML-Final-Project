import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score, roc_auc_score, root_mean_squared_error, f1_score,
    roc_curve, confusion_matrix, ConfusionMatrixDisplay, r2_score
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

    Returns:
        tuple: (trained_model, metrics_dict)
        
    metrics_dict contains:
        - accuracy: mean accuracy across folds
        - auc: mean AUC across folds  
        - f1_score: mean F1 score across folds
        - accuracy_std: std of accuracy across folds
        - auc_std: std of AUC across folds
        - f1_std: std of F1 score across folds
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    acc_scores, auc_scores, f1_scores = [], [], []
    all_fprs, all_tprs, all_aucs = [], [], []
    
    all_y_true = []
    all_y_pred = []
    
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
            eval_metric='logloss',
            early_stopping_rounds=10, 
        )

        print(f"[Fold {i+1}] XGBClassifier params:", model.get_params())  
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc_scores.append(accuracy_score(y_test, y_pred))
        auc_score = roc_auc_score(y_test, y_proba)
        auc_scores.append(auc_score)
        f1_scores.append(f1_score(y_test, y_pred))

        if plot:
            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)

            fpr, tpr, _ = roc_curve(y_test, y_proba)
            all_fprs.append(fpr)
            all_tprs.append(tpr)
            all_aucs.append(auc_score)

        last_model = model 

    metrics = {
        'accuracy': np.mean(acc_scores),
        'auc': np.mean(auc_scores),
        'f1_score': np.mean(f1_scores),
        'accuracy_std': np.std(acc_scores),
        'auc_std': np.std(auc_scores),
        'f1_std': np.std(f1_scores)
    }

    print(f"XGBoost Classification (KFold={n_splits}):")
    print(f"  Accuracy: {metrics['accuracy']:.4f} ± {metrics['accuracy_std']:.4f}")
    print(f"  AUC:      {metrics['auc']:.4f} ± {metrics['auc_std']:.4f}")
    print(f"  F1-Score: {metrics['f1_score']:.4f} ± {metrics['f1_std']:.4f}")

    if plot and all_fprs:
        if all_y_true and all_y_pred:
            cm_merged = confusion_matrix(all_y_true, all_y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm_merged)
            plt.figure(figsize=(8, 6))
            disp.plot()
            plt.title(f"Merged Confusion Matrix - All {n_splits} Folds")
            plt.tight_layout()
            plt.show()
        
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

        _plot_feature_importance(last_model, feature_names)

        if SHAP_AVAILABLE:
            print("Generating SHAP summary plot...") 
            X_sample = X if X.shape[0] <= 5000 else X[np.random.choice(X.shape[0], 5000, replace=False)]
            explainer = shap.Explainer(last_model, X_sample)
            shap_values = explainer(X_sample)
            shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=True)
        else:
            print("SHAP not available, skipping explanation plot.")

    if save_model and last_model is not None:
        joblib.dump(last_model, "xgboost_classifier_model.joblib")
        print("Saved: xgboost_classifier_model.joblib")

    return last_model, metrics

def train_xgboost_regressor(X, y, feature_names=None, n_splits=5, plot=True, save_model=True):
    """
    Train an XGBoost regressor with K-Fold CV, plots, and model saving.

    Returns:
        tuple: (trained_model, metrics_dict)
        
    metrics_dict contains:
        - rmse: mean RMSE across folds
        - r2: mean R² across folds
        - mse: mean MSE across folds
        - rmse_std: std of RMSE across folds
        - r2_std: std of R² across folds
        - mse_std: std of MSE across folds
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmse_scores, r2_scores, mse_scores = [], [], []
    preds_all, trues_all = [], []
    last_model = None

    for i, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=10,
        )

        print(f"[Fold {i+1}] XGBRegressor params:", model.get_params())

        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        y_pred = model.predict(X_test)
        
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mse = rmse ** 2 
        
        rmse_scores.append(rmse)
        r2_scores.append(r2)
        mse_scores.append(mse)

        preds_all.extend(y_pred)
        trues_all.extend(y_test)
        last_model = model

    metrics = {
        'rmse': np.mean(rmse_scores),
        'r2': np.mean(r2_scores),
        'mse': np.mean(mse_scores),
        'rmse_std': np.std(rmse_scores),
        'r2_std': np.std(r2_scores),
        'mse_std': np.std(mse_scores)
    }

    print(f"XGBoost Regression (KFold={n_splits}):")
    print(f"  RMSE: {metrics['rmse']:.4f} ± {metrics['rmse_std']:.4f}")
    print(f"  R²:   {metrics['r2']:.4f} ± {metrics['r2_std']:.4f}")
    print(f"  MSE:  {metrics['mse']:.4f} ± {metrics['mse_std']:.4f}")

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
        print("Saved: xgboost_regressor_model.joblib")

    return last_model, metrics
