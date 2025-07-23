"""
Professional Depression Classification System

A comprehensive machine learning pipeline for depression classification using multiple algorithms
with hyperparameter optimization, data balancing, and model interpretability.

Author: ML Team
Version: 2.0.0
Date: 2025-07-23
"""

import argparse
import json
import logging
import os
import sys
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from imblearn.combine import SMOTEENN
from scipy.stats import randint, uniform
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, brier_score_loss, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('depression_classification.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration class for model parameters."""
    name: str
    model_class: type
    param_grid: Dict[str, Any]
    random_state: int = 42


@dataclass
class ExperimentConfig:
    """Configuration class for experiment settings."""
    data_path: Optional[str] = None
    target_column: str = "Depression"
    test_size: float = 0.2
    cv_folds: int = 5
    n_iter: int = 10
    random_state: int = 42
    output_dir: str = "outputs/models/randomized_search"
    scoring_metric: str = "accuracy"
    balance_data: bool = True
    perform_shap: bool = True
    
    # Model configurations
    models: List[ModelConfig] = field(default_factory=lambda: [
        ModelConfig(
            name="Logistic Regression",
            model_class=LogisticRegression,
            param_grid={
                "C": uniform(0.1, 100.0),
                "solver": ['lbfgs'],
                "penalty": ['l2'],
                "class_weight": ['balanced', None],
                "max_iter": [5000]
            }
        ),
        ModelConfig(
            name="Gradient Boosting",
            model_class=GradientBoostingClassifier,
            param_grid={
                "n_estimators": randint(100, 501),
                "learning_rate": uniform(0.01, 0.19),
                "max_depth": randint(3, 9),
                "min_samples_split": randint(2, 11),
                "min_samples_leaf": randint(1, 5),
                "subsample": uniform(0.8, 0.2)
            }
        ),
        ModelConfig(
            name="SVM",
            model_class=SVC,
            param_grid={
                "C": uniform(0.1, 100),
                "kernel": ['rbf', 'linear'],
                "gamma": ['scale', 'auto', 0.001, 0.01, 0.1],
                "class_weight": ['balanced', None],
                "probability": [True]
            }
        )
    ])


class DataValidator:
    """Validates input data for the classification pipeline."""
    
    @staticmethod
    def validate_data(data: pd.DataFrame, target_column: str) -> bool:
        """
        Validates the input DataFrame.
        
        Args:
            data: Input DataFrame
            target_column: Name of the target column
            
        Returns:
            bool: True if data is valid
            
        Raises:
            ValueError: If validation fails
        """
        if data.empty:
            raise ValueError("Input data is empty")
            
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
            
        if data[target_column].isnull().all():
            raise ValueError(f"Target column '{target_column}' contains only null values")
            
        if len(data[target_column].unique()) < 2:
            raise ValueError(f"Target column '{target_column}' must have at least 2 unique values")
            
        logger.info(f"Data validation passed: {data.shape[0]} samples, {data.shape[1]} features")
        return True


class DataPreprocessor:
    """Handles data preprocessing operations."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.label_encoders: Dict[str, LabelEncoder] = {}
        
    def preprocess(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocesses the input data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Tuple of processed features and target
        """
        logger.info("Starting data preprocessing...")
        
        # Separate features and target
        X = data.drop([self.config.target_column], axis=1).copy()
        y = data[self.config.target_column].copy()
        
        # Handle missing values
        X = self._handle_missing_values(X)
        
        # Encode categorical features
        X = self._encode_categorical_features(X)
        
        logger.info(f"Preprocessing completed: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    
    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handles missing values in the dataset."""
        missing_counts = X.isnull().sum()
        if missing_counts.sum() > 0:
            logger.info(f"Found missing values in {(missing_counts > 0).sum()} columns")
            
            # Fill numerical columns with mean
            numerical_cols = X.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if X[col].isnull().sum() > 0:
                    X[col].fillna(X[col].mean(), inplace=True)
                    
            # Fill categorical columns with mode
            categorical_cols = X.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if X[col].isnull().sum() > 0:
                    X[col].fillna(X[col].mode()[0], inplace=True)
                    
        return X
    
    def _encode_categorical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encodes categorical features using LabelEncoder."""
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        if len(categorical_cols) > 0:
            logger.info(f"Encoding {len(categorical_cols)} categorical features")
            
            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                self.label_encoders[col] = le
                
        return X


class ModelTrainer:
    """Handles model training and evaluation."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.best_models: Dict[str, Any] = {}
        self.best_params: Dict[str, Dict] = {}
        self.metrics: Dict[str, Dict] = {}
        
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Trains all configured models.
        
        Args:
            X: Feature matrix
            y: Target vector
        """
        logger.info("Starting model training and evaluation...")
        
        cv = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state
        )
        
        for model_config in self.config.models:
            logger.info(f"Training {model_config.name}...")
            
            # Create model instance
            model = model_config.model_class(random_state=self.config.random_state)
            
            # Perform hyperparameter optimization
            search = RandomizedSearchCV(
                model,
                model_config.param_grid,
                n_iter=self.config.n_iter,
                cv=cv,
                scoring=self.config.scoring_metric,
                n_jobs=-1,
                random_state=self.config.random_state,
                verbose=0
            )
            
            search.fit(X, y)
            
            # Store best model and parameters
            self.best_models[model_config.name] = search.best_estimator_
            self.best_params[model_config.name] = search.best_params_
            
            # Evaluate model
            metrics = self._evaluate_model(search.best_estimator_, X, y, cv)
            self.metrics[model_config.name] = metrics
            
            logger.info(f"{model_config.name} - Best score: {search.best_score_:.4f}")
            
    def _evaluate_model(self, model: Any, X: pd.DataFrame, y: pd.Series, 
                       cv: StratifiedKFold) -> Dict[str, float]:
        """
        Evaluates a model using cross-validation.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target vector
            cv: Cross-validation strategy
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {
            'accuracy': [], 'roc_auc': [], 'brier_score': [],
            'recall': [], 'precision': [], 'f1': [],
            'ppv': [], 'npv': []
        }
        
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate confusion matrix components
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            
            # Store metrics
            metrics['accuracy'].append(accuracy_score(y_test, y_pred))
            metrics['roc_auc'].append(roc_auc_score(y_test, y_pred_proba))
            metrics['brier_score'].append(brier_score_loss(y_test, y_pred_proba))
            metrics['recall'].append(recall_score(y_test, y_pred))
            metrics['precision'].append(precision_score(y_test, y_pred))
            metrics['f1'].append(f1_score(y_test, y_pred))
            metrics['ppv'].append(ppv)
            metrics['npv'].append(npv)
        
        # Return mean metrics
        return {metric: np.mean(values) for metric, values in metrics.items()}


class ResultsManager:
    """Manages model results, visualization, and saving."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        
    def create_visualizations(self, metrics: Dict[str, Dict]) -> None:
        """Creates and saves visualization plots."""
        logger.info("Creating visualizations...")
        
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Model comparison plot
        self._create_comparison_plot(metrics, output_dir)
        
    def _create_comparison_plot(self, metrics: Dict[str, Dict], output_dir: Path) -> None:
        """Creates model comparison plot."""
        plt.figure(figsize=(15, 10))
        
        metrics_to_plot = ['accuracy', 'roc_auc', 'f1', 'precision', 'recall', 'ppv', 'npv', 'brier_score']
        x = np.arange(len(metrics))
        width = 0.11
        
        for i, metric in enumerate(metrics_to_plot):
            offset = width * i
            values = [metrics[model][metric] for model in metrics.keys()]
            plt.bar(x + offset, values, width, label=metric.replace('_', ' ').title())
        
        plt.xlabel('Models')
        plt.ylabel('Scores')
        plt.title('Model Performance Comparison')
        plt.xticks(x + width * 3.5, metrics.keys(), rotation=45)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        
        plt.savefig(output_dir / 'model_comparison.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        logger.info("Model comparison plot saved")
        
    def save_results(self, best_models: Dict, best_params: Dict, metrics: Dict) -> None:
        """Saves results to files."""
        logger.info("Saving results...")
        
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find best model
        best_model_name = max(metrics.items(), key=lambda x: x[1]['accuracy'])[0]
        
        # Save detailed results
        results = {
            'best_model': best_model_name,
            'best_accuracy': metrics[best_model_name]['accuracy'],
            'all_metrics': metrics,
            'best_parameters': best_params,
            'experiment_config': {
                'cv_folds': self.config.cv_folds,
                'n_iter': self.config.n_iter,
                'random_state': self.config.random_state
            }
        }
        
        with open(output_dir / 'results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
            
        # Save human-readable report
        self._save_text_report(best_model_name, metrics, best_params, output_dir)
        
        logger.info(f"Results saved to {output_dir}")
        
    def _save_text_report(self, best_model_name: str, metrics: Dict, 
                         best_params: Dict, output_dir: Path) -> None:
        """Saves human-readable text report."""
        with open(output_dir / 'model_results.txt', 'w', encoding='utf-8') as f:
            f.write("DEPRESSION CLASSIFICATION RESULTS\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"BEST MODEL: {best_model_name}\n")
            f.write(f"Accuracy: {metrics[best_model_name]['accuracy']:.4f}\n")
            f.write("-" * 50 + "\n\n")
            
            f.write("ALL MODEL RESULTS:\n")
            for model_name, model_metrics in metrics.items():
                f.write(f"\n{model_name}:\n")
                for metric_name, value in model_metrics.items():
                    f.write(f"  {metric_name}: {value:.4f}\n")
                f.write("-" * 30 + "\n")
            
            f.write("\nBEST PARAMETERS:\n")
            for model_name, params in best_params.items():
                f.write(f"\n{model_name}:\n")
                for param, value in params.items():
                    f.write(f"  {param}: {value}\n")


class SHAPAnalyzer:
    """Handles SHAP analysis for model interpretability."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        
    def analyze(self, best_models: Dict, metrics: Dict, X: pd.DataFrame) -> None:
        """
        Performs SHAP analysis on the best model.
        
        Args:
            best_models: Dictionary of trained models
            metrics: Model performance metrics
            X: Feature matrix
        """
        if not self.config.perform_shap:
            return
            
        logger.info("Starting SHAP analysis...")
        
        # Find best model
        best_model_name = max(metrics.items(), key=lambda x: x[1]['accuracy'])[0]
        best_model = best_models[best_model_name]
        
        # Only perform SHAP for tree-based models
        if isinstance(best_model, GradientBoostingClassifier):
            self._perform_tree_shap(best_model, X, best_model_name)
        else:
            logger.info(f"SHAP analysis not supported for {best_model_name}")
            
    def _perform_tree_shap(self, model: Any, X: pd.DataFrame, model_name: str) -> None:
        """Performs SHAP analysis for tree-based models."""
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X.sample(min(1000, len(X)), random_state=42))
            
            output_dir = Path(self.config.output_dir)
            
            # Summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X.sample(min(1000, len(X)), random_state=42), show=False)
            plt.tight_layout()
            plt.savefig(output_dir / 'shap_summary.png', bbox_inches='tight', dpi=300)
            plt.close()
            
            # Bar plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X.sample(min(1000, len(X)), random_state=42), 
                            plot_type="bar", show=False)
            plt.tight_layout()
            plt.savefig(output_dir / 'shap_importance_bar.png', bbox_inches='tight', dpi=300)
            plt.close()
            
            # Save feature importance
            feature_importance = np.abs(shap_values).mean(0)
            importance_dict = dict(zip(X.columns, feature_importance))
            sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            
            with open(output_dir / 'feature_importance.txt', 'w', encoding='utf-8') as f:
                f.write("FEATURE IMPORTANCE (SHAP)\n")
                f.write("=" * 30 + "\n")
                for feature, importance in sorted_features:
                    f.write(f"{feature}: {importance:.4f}\n")
                    
            logger.info("SHAP analysis completed")
            
        except Exception as e:
            logger.error(f"SHAP analysis failed: {str(e)}")


class DepressionClassifier:
    """
    Professional Depression Classification System.
    
    A comprehensive machine learning pipeline for depression classification
    using multiple algorithms with hyperparameter optimization.
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.data_validator = DataValidator()
        self.preprocessor = DataPreprocessor(config)
        self.trainer = ModelTrainer(config)
        self.results_manager = ResultsManager(config)
        self.shap_analyzer = SHAPAnalyzer(config)
        
    def run_pipeline(self) -> bool:
        """
        Runs the complete classification pipeline.
        
        Returns:
            bool: True if pipeline completed successfully
        """
        try:
            logger.info("Starting Depression Classification Pipeline...")
            
            # Load and validate data
            data = self._load_data()
            self.data_validator.validate_data(data, self.config.target_column)
            
            # Preprocess data
            X, y = self.preprocessor.preprocess(data)
            
            # Balance data if requested
            if self.config.balance_data:
                X, y = self._balance_data(X, y)
            
            # Train models
            self.trainer.train_models(X, y)
            
            # Create visualizations
            self.results_manager.create_visualizations(self.trainer.metrics)
            
            # Save results
            self.results_manager.save_results(
                self.trainer.best_models,
                self.trainer.best_params,
                self.trainer.metrics
            )
            
            # Perform SHAP analysis
            self.shap_analyzer.analyze(self.trainer.best_models, self.trainer.metrics, X)
            
            # Print summary
            self._print_summary()
            
            logger.info("Pipeline completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            return False
    
    def _load_data(self) -> pd.DataFrame:
        """Loads data from file."""
        if self.config.data_path is None:
            print("Please specify the data file path:")
            print("Example: 'data/depression.xlsx' or full path")
            self.config.data_path = input("Data file path: ")
        
        logger.info(f"Loading data from: {self.config.data_path}")
        
        try:
            if self.config.data_path.endswith('.xlsx'):
                data = pd.read_excel(self.config.data_path)
            elif self.config.data_path.endswith('.csv'):
                data = pd.read_csv(self.config.data_path)
            else:
                raise ValueError("Unsupported file format. Use .xlsx or .csv")
                
            logger.info(f"Data loaded successfully: {data.shape}")
            return data
            
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {self.config.data_path}")
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def _balance_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Balances the dataset using SMOTEENN."""
        logger.info("Balancing dataset with SMOTEENN...")
        
        class_counts = y.value_counts()
        imbalance_ratio = class_counts.max() / class_counts.min()
        logger.info(f"Original class distribution: {dict(class_counts)}")
        logger.info(f"Imbalance ratio: {imbalance_ratio:.2f}")
        
        smote_enn = SMOTEENN(random_state=self.config.random_state)
        X_resampled, y_resampled = smote_enn.fit_resample(X, y)
        
        new_counts = pd.Series(y_resampled).value_counts()
        logger.info(f"Balanced class distribution: {dict(new_counts)}")
        
        return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)
    
    def _print_summary(self) -> None:
        """Prints pipeline summary."""
        best_model_name = max(self.trainer.metrics.items(), key=lambda x: x[1]['accuracy'])[0]
        best_accuracy = self.trainer.metrics[best_model_name]['accuracy']
        
        print("\n" + "=" * 60)
        print("DEPRESSION CLASSIFICATION RESULTS SUMMARY")
        print("=" * 60)
        print(f"Best Model: {best_model_name}")
        print(f"Best Accuracy: {best_accuracy:.4f}")
        print(f"Results saved to: {self.config.output_dir}")
        print("=" * 60)


def create_config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    """Creates experiment configuration from command line arguments."""
    return ExperimentConfig(
        data_path=args.data_path,
        target_column=args.target_column,
        cv_folds=args.cv_folds,
        n_iter=args.n_iter,
        random_state=args.random_state,
        output_dir=args.output_dir,
        balance_data=args.balance_data,
        perform_shap=args.perform_shap
    )


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Professional Depression Classification System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--data-path", "-d",
        type=str,
        help="Path to the dataset file (.xlsx or .csv)"
    )
    
    parser.add_argument(
        "--target-column", "-t",
        type=str,
        default="Depression",
        help="Name of the target column"
    )
    
    parser.add_argument(
        "--cv-folds", "-cv",
        type=int,
        default=5,
        help="Number of cross-validation folds"
    )
    
    parser.add_argument(
        "--n-iter", "-n",
        type=int,
        default=10,
        help="Number of parameter settings sampled"
    )
    
    parser.add_argument(
        "--random-state", "-r",
        type=int,
        default=42,
        help="Random state for reproducibility"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="outputs/models/randomized_search",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--no-balance",
        action="store_false",
        dest="balance_data",
        help="Disable data balancing"
    )
    
    parser.add_argument(
        "--no-shap",
        action="store_false",
        dest="perform_shap",
        help="Disable SHAP analysis"
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = create_config_from_args(args)
    
    # Run pipeline
    classifier = DepressionClassifier(config)
    success = classifier.run_pipeline()
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()