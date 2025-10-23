"""
Model evaluation module for Air Quality ML Pipeline.

This module provides core evaluation functionality used by the package.
For detailed evaluation with visualizations, see utils.evaluation_utils.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold,KFold
from sklearn.metrics import mean_absolute_error, r2_score,root_mean_squared_error
from sklearn.base import clone

# TODO Import GridSearchCV for hyperparameter optimization (Workshop 3)
from sklearn.model_selection import GridSearchCV
# TODO Import MLflow (Workshop 4)
import mlflow
from utils.config import N_SPLITS, RANDOM_STATE
from utils.logger import get_logger, LogLevel


class Evaluator:
    """
    Core evaluator for air quality prediction models.
    
    This class handles basic model evaluation including cross-validation
    and metrics calculation used by the package components.
    """
    
    def __init__(self):
        """Initialize the evaluator."""
        pass
    
    def calculate_metrics(self, y_true, y_pred):
        """
        Calculate comprehensive regression metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            Dictionary with calculated metrics
        """
        # TODO Calculate comprehensive regression metrics        
        rmse = root_mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # Create metrics dictionary
        metrics = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        return metrics
    
    def cross_validate_model(self, model, X, y, groups=None):
        """
        Perform cross-validation using GroupKFold.
        
        This method uses GroupKFold to ensure entire cities are either in training 
        OR validation, never both, preventing data leakage.
        
        Args:
            model: Scikit-learn model to evaluate
            X: Feature matrix
            y: Target variable
            groups: Grouping variable for GroupKFold (e.g., cities)
            
        Returns:
            Dictionary with cross-validation results
        """
        logger = get_logger()
        logger.info(f"Cross-validating {model.__class__.__name__}...", LogLevel.NORMAL)
        
        # TODO Set up GroupKFold cross-validation
        if groups is not None:
            cv = GroupKFold(n_splits=N_SPLITS)
        else:
            cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

        fold_results = []

        # TODO Perform cross-validation enumerating folds
        # Split data
        X_work = X.copy()
        X_work["fold"] = -1   # initialize with -1
        fold_col_idx = X_work.columns.get_loc("fold")
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups=groups)):
            X_work.iloc[val_idx, fold_col_idx] = fold

        for fold in range(N_SPLITS):

            # Train model
            model = clone(model)
    
            # Create train/validation split for current fold
            train_mask = X_work['fold'] != fold
            val_mask = X_work['fold'] == fold
            X_train_fold = X_work[train_mask].drop(columns=["fold"])
            y_train_fold = y[train_mask]
            X_val_fold = X_work[val_mask].drop(columns=["fold"])
            y_val_fold = y[val_mask]

            model.fit(X_train_fold, y_train_fold)
            
            # Predict
            val_preds = model.predict(X_val_fold)
            
            # Calculate metrics and append to fold_results
            metrics = self.calculate_metrics(y_val_fold, val_preds)
            fold_results.append(metrics)
            rmse = metrics['rmse']
            r2 = metrics['r2']
            # Logging
            logger.info(f"Fold {fold}: rmse: {rmse:.4f}, r2_score: {r2:.4f}")
        
        # Aggregate results
        cv_results = {}
        # Enumerate metrics and calculate mean/std across folds
        for metric in fold_results[0].keys():
            values = [fold[metric] for fold in fold_results]
            cv_results[f'{metric}_mean'] = np.mean(values)
            cv_results[f'{metric}_std'] = np.std(values)
        
        # TODO Add MLflow cross-validation metrics logging (Workshop 4)  
            
                 
            # Log cross-validation results (metrics only - must be numeric)
            cv_metrics = {
                'cv_rmse_mean': rmse_mean,
                'cv_rmse_std': rmse_std,
                'cv_mae_mean': mae_mean, # Assuming MAE is calculated
            }
            # Add additional CV metadata (metrics only - must be numeric)
            cv_metrics.update({
                'cv_n_folds': int(cv_strategy) if isinstance(cv_strategy, int) else 0,
                'cv_n_samples': X.shape[0],
            })
            mlflow.log_metrics(cv_metrics)
            
            # Log strategy as parameter (strings allowed in parameters)

        # Logging
        if logger.level >= LogLevel.NORMAL:
            print(f"  Average: RMSE={cv_results['rmse_mean']:.3f}±{cv_results['rmse_std']:.3f}")
        
        return cv_results
    
    
    def hyperparameter_optimization_cv(self, model, param_grid, X, y, groups=None):
        """
        Perform hyperparameter optimization using GridSearchCV with geographic cross-validation.
        
        This method combines GridSearchCV with GroupKFold to ensure that entire cities
        are either in training OR validation during hyperparameter search, preventing data leakage.
        
        Args:
            model: Scikit-learn model to optimize
            param_grid: Dictionary of hyperparameters to search
            X: Feature matrix
            y: Target variable
            groups: Grouping variable for GroupKFold (e.g., cities)
            
        Returns:
            Tuple of (best_model, best_params, best_score)
        """
        logger = get_logger()
        logger.info(f"Optimizing hyperparameters for {model.__class__.__name__}...", LogLevel.NORMAL)
        
        # TODO Add hyperparameter optimization with geographic cross-validation (Workshop 3)
        # Set up cross-validation strategy
        if groups is not None:
            cv = GroupKFold(n_splits=N_SPLITS)
            cv_split = cv.split(X, y, groups=groups)
            logger.info(f"Using GroupKFold (n_splits={N_SPLITS}) for geographic CV")
        else:
            cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
            cv_split = cv.split(X, y)
            logger.info(f"Using KFold (n_splits={N_SPLITS}, shuffle=True)")
        
        # Configure GridSearchCV with geographic cross-validation
        verbose = 1 if logger.level >= LogLevel.VERBOSE else 0
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring="neg_root_mean_squared_error",
            cv=cv_split,
            n_jobs=-1,
            refit=True,
            return_train_score=True,
            verbose=verbose,
        )

        # Fit GridSearchCV
        grid_search.fit(X, y)
        
        # Extract results (GridSearchCV returns negative RMSE, convert to positive)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = -float(grid_search.best_score_)
        
        # Logging
        logger.success(f"Best RMSE: {best_score:.3f}")
        if logger.level >= LogLevel.NORMAL:
            print(f"  Best parameters: {best_params}")
        
        return best_model, best_params, best_score

