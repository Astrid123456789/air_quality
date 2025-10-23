"""
Feature engineering module for Air Quality ML Pipeline.

This module handles feature extraction and selection.
Students need to complete the TODO sections.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression
from utils.config import (
    TEMPORAL_FEATURES, N_FEATURES_SELECTKBEST, N_FEATURES_RFE,
    DATE_COL, LATITUDE_COL, LONGITUDE_COL, TARGET_COL
)
from utils.logger import get_logger, LogLevel

# TODO Import MLflow (Workshop 4)
import mlflow

class FeatureEngineer:
    """
    Feature engineer for air quality prediction.
    
    Handles temporal feature extraction, geographic feature creation,
    categorical encoding, and feature selection.
    """
    print("test")
    print("test")

    
    def __init__(self):
        """Initialize the feature engineer."""
        self.feature_columns = None
        self.label_encoders = {}
        self.feature_selector = None
        self.selected_features = None
    
    def extract_temporal_features(self, df):
        """
        Extract temporal features from datetime column.
        
        Air quality varies with seasonal changes, weekly patterns,
        and daily cycles. These features help models understand when
        pollution levels are typically higher or lower.
        
        Args:
            df: DataFrame with date column
            
        Returns:
            DataFrame with additional temporal features
        """
        logger = get_logger()
        logger.substep("Extracting Temporal Features")
        
        # TODO Copy the DataFrame into df_features to avoid modifying the original
        df_features = df.copy()
        # TODO Ensure the date column is datetime type for proper extraction
        df_features['date'] = pd.to_datetime(df_features['date'])

        # TODO Extract basic temporal components that affect air quality
        df_features['year'] = df_features['date'].dt.year
        df_features['month'] = df_features['date'].dt.month
        df_features['day'] = df_features['date'].dt.day
        df_features['quarter'] = df_features['date'].dt.quarter
        df_features['week'] = df_features['date'].dt.isocalendar().week
        df_features['dayofweek'] = df_features['date'].dt.dayofweek
        df_features['is_weekend'] = df_features['dayofweek'].isin([5, 6]).astype(int)
        df_features['is_month_start'] = df_features['date'].dt.is_month_start.astype(int)
        df_features['is_month_end'] = df_features['date'].dt.is_month_end.astype(int)
        # Logging
        logger.success("Temporal features extracted")

        return df_features
    
    def extract_geographic_features(self, df):
        """
        Extract geographic features from coordinate data.
        
        Different monitoring locations have unique pollution characteristics
        based on their surroundings. This creates location identifiers to
        help models learn location-specific patterns.
        
        Args:
            df: DataFrame with latitude and longitude columns
            
        Returns:
            DataFrame with additional geographic features
        """
        logger = get_logger()
        logger.substep("Extracting Geographic Features")
        
        # TODO Copy the DataFrame into df_features to avoid modifying the original
        df_features=df.copy()
        # TODO Create a unique location identifier by combining coordinates
        # This allows the model to learn location-specific patterns
        df_features['location'] = (df_features['site_latitude'].astype(str) + '_' + 
                        df['site_longitude'].astype(str))
        # Logging
        n_locations = df_features['location'].nunique()
        logger.success(f"Created location identifiers for {n_locations} unique locations")
        
        return df_features
    
    def encode_categorical_features(self, train_df, test_df, categorical_columns=None):
        """
        Encode categorical features for machine learning models.
        
        Machine learning algorithms require numerical inputs. This converts
        categorical variables to numerical format while ensuring consistency
        between training and test datasets.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame  
            categorical_columns: List of columns to encode (auto-detect if None)
            
        Returns:
            Tuple of (encoded_train_df, encoded_test_df)
        """
        logger = get_logger()
        logger.substep("Encoding Categorical Features")
        
        # TODO Combine datasets to ensure consistent encoding across train/test
        # This prevents issues where test set has categories not seen in training
        train_df = train_df.copy()
        test_df = test_df.copy()
        combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)
        combined[DATE_COL] = pd.to_datetime(combined[DATE_COL], errors="coerce")
        # TODO Initialize label encoder for consistent categorical-to-numerical conversion
        encoders = {}
        # TODO Identify categorical columns that need encoding
        categorical_columns = [col for col in combined.columns 
                       if combined[col].dtype == 'object' 
                       and col != 'date' 
                       and col != TARGET_COL]

        combined[DATE_COL] = combined[DATE_COL].view("int64")
    
        print("Categorical columns to encode:", categorical_columns)
        # These are high-cardinality categories (many unique values)
        # TODO Apply label encoding: convert each unique category to a unique integer
        for col in categorical_columns:
            le = LabelEncoder()
            combined[col] = le.fit_transform(combined[col].astype(str))
            encoders[col] = le
            logger.info(f"Encoded {col}: {len(le.classes_)} unique values")

        # TODO Split back into separate train and test datasets
        train_encoded = combined.iloc[:len(train_df)].copy()
        test_encoded = combined.iloc[len(train_df):].copy()
        # Logging
        logger.success("Categorical encoding completed")

        return train_encoded, test_encoded
    
    def select_features_selectkbest(self, X, y, k=15):
        """
        Select features using SelectKBest with f_regression.
        
        Args:
            X: Feature matrix
            y: Target variable
            k: Number of features to select
            
        Returns:
            List of selected feature names
        """
        logger = get_logger()
        logger.substep("Feature Selection - SelectKBest")
        
        # TODO Initialize and fit the selector
        selector_kbest = SelectKBest(score_func=f_regression, k=k)
        selector_kbest.fit(X, y)

        # TODO Get selected feature names and their scores
        selected_features_mask = selector_kbest.get_support()
        feature_scores = selector_kbest.scores_
        selected_features_kbest = X.columns[selected_features_mask]

        # TODO Create a summary DataFrame for selected features only
        feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Score': feature_scores,
        'Selected': selected_features_mask
    }).sort_values(by='Score', ascending=False)
        selected_df = feature_importance_df[feature_importance_df['Selected']]

        # Logging
        logger.info(f"Top {k} features selected by SelectKBest:", LogLevel.NORMAL)
        if logger.level >= LogLevel.NORMAL:
            print(selected_df)
        
        # Store selector and features
        self.feature_selector = selector_kbest
        self.selected_features = selected_features_kbest.tolist()
        
        return selected_features_kbest.tolist()
    
    def select_features_rfe(self, X, y, rfe_features=15):
        """
        Select features using Recursive Feature Elimination.
        
        Args:
            X: Feature matrix
            y: Target variable
            rfe_features: Number of features to select
            
        Returns:
            List of selected feature names
        """
        logger = get_logger()
        logger.substep("Feature Selection - RFE")
        
        # TODO Initialize RFE with LinearRegression as the estimator
        estimator = LinearRegression()
        selector_rfe = RFE(estimator=estimator, n_features_to_select=rfe_features)

        # TODO Fit RFE and transform features
        selector_rfe.fit(X, y)

        # TODO Get selected features and their rankings
        selected_features_mask = selector_rfe.support_
        selected_features_rfe = X.columns[selected_features_mask]

        # Logging
        logger.info(f"Top {rfe_features} features selected by RFE:", LogLevel.NORMAL)
        if logger.level >= LogLevel.NORMAL:
            rfe_df = pd.DataFrame({
                'Feature': selected_features_rfe,
                'Coefficient': selector_rfe.estimator_.coef_
            }).sort_values('Coefficient', key=abs, ascending=False)
            print(rfe_df)
        
        # Store selector and features in instance variables
        self.feature_selector = selector_rfe
        self.selected_features = selected_features_rfe.tolist()
        
        return selected_features_rfe.tolist()
    
    def get_feature_columns(self, df, exclude_target=True, exclude_metadata=True):
        """
        Get list of columns that can be used as features.
        
        Args:
            df: DataFrame to analyze
            exclude_target: Whether to exclude target column
            exclude_metadata: Whether to exclude metadata columns
            
        Returns:
            List of feature column names
        """
        # Start with all columns
        feature_cols = df.columns.tolist()
        
        # Columns to exclude
        exclude_cols = []
        
        # Exclude target column if exclude_target is True and TARGET_COL is defined
        if exclude_target and TARGET_COL in feature_cols:
            exclude_cols.append(TARGET_COL)
        
        # Exclude metadata columns if exclude_metadata is True
        # Metadata columns are those that do not contribute to the model
        # but are useful for understanding the data context
        if exclude_metadata:
            # Common metadata columns to exclude (but keep date as it's encoded as feature)
            metadata_cols = ['id', 'folds', 'site_id', 'country', 'city', 'site_latitude', 'site_longitude']
            exclude_cols.extend([col for col in metadata_cols if col in feature_cols])
        
        # Remove excluded columns
        feature_cols = [col for col in feature_cols if col not in exclude_cols]
        
        # Store features in instance variables
        self.feature_columns = feature_cols
        return feature_cols
    
    def extract_all_features(self, train_df, test_df):
        """
        Extract all features from train and test data.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            
        Returns:
            Tuple of (train_with_features, test_with_features)
        """
        logger = get_logger()
        logger.info("Extracting all features...")
        
        # Extract temporal features
        train_features = self.extract_temporal_features(train_df)
        test_features = self.extract_temporal_features(test_df)
        
        # Extract geographic features
        train_features = self.extract_geographic_features(train_features)
        test_features = self.extract_geographic_features(test_features)
        
        # Encode categorical features
        train_features, test_features = self.encode_categorical_features(
            train_features, test_features
        )
        
        # Logging
        logger.success("All features extracted")
        
        return train_features, test_features
    
    def select_best_features(self, train_df, method='selectkbest', n_features=None):
        """
        Select the best features using specified method.
        
        Args:
            train_df: Training DataFrame with features and target
            method: 'selectkbest' or 'rfe'
            n_features: Number of features to select
            
        Returns:
            List of selected feature names
        """
        # Get feature columns
        feature_cols = self.get_feature_columns(train_df)
        
        X = train_df[feature_cols]
        y = train_df[TARGET_COL]

        # TODO Add MLflow feature selection logging (Workshop 4)       
            # Log feature selection parameters
            mlflow.log_params({
                'Selection_method': method,
                'Total_features_available_before_selection': X.shape[1],
                'Number_of_features_to_select': n_features if n_features is not None else (
                    N_FEATURES_SELECTKBEST if method == 'selectkbest' else N_FEATURES_RFE
                )
                'target_column': TARGET_COL,

            })
        
        if method == 'selectkbest':
            return self.select_features_selectkbest(X, y, n_features)
        elif method == 'rfe':
            return self.select_features_rfe(X, y, n_features)
        else:
            raise ValueError(f"Unknown selection method: {method}")
