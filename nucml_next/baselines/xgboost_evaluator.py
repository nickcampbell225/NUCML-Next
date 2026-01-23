"""
XGBoost Evaluator
=================

Enhanced baseline using gradient boosting.

Compared to Decision Trees:
    - Better accuracy (ensemble of trees)
    - Still exhibits staircase effect (milder)
    - Still poor extrapolation
    - Can't learn smooth resonance curves

Educational Purpose:
    Show that even state-of-the-art classical ML can't match
    physics-informed deep learning for smooth predictions.
"""

from typing import Dict, Any, Optional, Tuple, Callable
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import joblib

try:
    from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False


class XGBoostEvaluator:
    """
    XGBoost baseline for cross-section prediction.

    Better than Decision Trees but still fundamentally limited:
    - Ensemble smooths out some steps
    - Feature importance reveals physics insights
    - But still can't guarantee smoothness or physical constraints
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        gamma: float = 0.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        min_child_weight: int = 1,
        random_state: int = 42,
    ):
        """
        Initialize XGBoost evaluator.

        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate (eta)
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of features
            gamma: Minimum loss reduction for split (regularization)
            reg_alpha: L1 regularization term on weights
            reg_lambda: L2 regularization term on weights
            min_child_weight: Minimum sum of instance weight needed in child
            random_state: Random seed
        """
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'gamma': gamma,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'min_child_weight': min_child_weight,
            'random_state': random_state,
            'objective': 'reg:squarederror',
            'tree_method': 'hist',  # Fast histogram-based method
        }

        self.model = xgb.XGBRegressor(**self.params)
        self.is_trained = False
        self.feature_columns = None
        self.metrics = {}

    def optimize_hyperparameters(
        self,
        df: pd.DataFrame,
        target_column: str = 'CrossSection',
        exclude_columns: Optional[list] = None,
        max_evals: int = 100,
        cv_folds: int = 3,
        test_size: float = 0.2,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Bayesian optimization (hyperopt).

        This finds the optimal n_estimators, max_depth, learning_rate, and other
        XGBoost parameters for the given dataset.

        Args:
            df: Training data (from NucmlDataset.to_tabular())
            target_column: Name of target column
            exclude_columns: Columns to exclude from features
            max_evals: Maximum number of hyperparameter evaluations
            cv_folds: Number of cross-validation folds
            test_size: Fraction of data for final test set
            verbose: Print optimization progress

        Returns:
            Dictionary with:
                - best_params: Optimized hyperparameters
                - best_score: Best cross-validation score
                - trials: Hyperopt trials object for analysis

        Example:
            >>> xgb_model = XGBoostEvaluator()
            >>> result = xgb_model.optimize_hyperparameters(df_train, max_evals=50)
            >>> print(f"Optimal trees: {result['best_params']['n_estimators']}")
            >>> # Now create new model with optimal params
            >>> xgb_optimal = XGBoostEvaluator(**result['best_params'])
            >>> xgb_optimal.train(df_train)
        """
        if not HYPEROPT_AVAILABLE:
            raise ImportError(
                "hyperopt is required for hyperparameter optimization.\n"
                "Install with: pip install hyperopt"
            )

        if verbose:
            print("\n" + "=" * 80)
            print("HYPERPARAMETER OPTIMIZATION - XGBoost")
            print("=" * 80)
            print(f"Dataset size: {len(df):,} samples")
            print(f"Max evaluations: {max_evals}")
            print(f"Cross-validation folds: {cv_folds}")
            print()

        # Prepare features and target
        if exclude_columns is None:
            exclude_columns = [target_column, 'Isotope', 'Reaction']

        numeric_columns = df.select_dtypes(include=[np.number, pd.SparseDtype]).columns
        feature_columns = [col for col in numeric_columns if col not in exclude_columns]

        # Handle sparse DataFrames
        X_df = df[feature_columns]
        sparse_columns = [col for col in X_df.columns if isinstance(X_df[col].dtype, pd.SparseDtype)]

        if len(sparse_columns) > 0:
            import scipy.sparse as sp
            try:
                if len(sparse_columns) == len(X_df.columns):
                    X = sp.csr_matrix(X_df.sparse.to_coo())
                else:
                    X = X_df.values
            except (AttributeError, ValueError):
                X = X_df.values
        else:
            X = X_df.values

        y = df[target_column].values

        # Log-transform target
        y_log = np.log10(y + 1e-10)

        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_log, test_size=test_size, random_state=42
        )

        # Define hyperparameter search space
        space = {
            'n_estimators': hp.quniform('n_estimators', 50, 500, 50),
            'max_depth': hp.quniform('max_depth', 3, 15, 1),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
            'subsample': hp.uniform('subsample', 0.6, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
            'gamma': hp.loguniform('gamma', np.log(1e-8), np.log(1.0)),
            'reg_alpha': hp.loguniform('reg_alpha', np.log(1e-8), np.log(1.0)),
            'reg_lambda': hp.loguniform('reg_lambda', np.log(1e-8), np.log(10.0)),
            'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
        }

        # Define objective function for hyperopt
        def objective(params):
            # Convert float hyperparameters to int where needed
            params['n_estimators'] = int(params['n_estimators'])
            params['max_depth'] = int(params['max_depth'])
            params['min_child_weight'] = int(params['min_child_weight'])

            # Create model with these hyperparameters
            model = xgb.XGBRegressor(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                learning_rate=params['learning_rate'],
                subsample=params['subsample'],
                colsample_bytree=params['colsample_bytree'],
                gamma=params['gamma'],
                reg_alpha=params['reg_alpha'],
                reg_lambda=params['reg_lambda'],
                min_child_weight=params['min_child_weight'],
                random_state=42,
                objective='reg:squarederror',
                tree_method='hist',
                n_jobs=-1,
                verbosity=0,
            )

            # Cross-validation score (negative MSE, since hyperopt minimizes)
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=cv_folds,
                scoring='neg_mean_squared_error',
                n_jobs=1  # XGBoost already uses n_jobs=-1
            )

            # Return mean CV score
            mean_cv_score = -cv_scores.mean()

            return {
                'loss': mean_cv_score,
                'status': STATUS_OK,
                'params': params,
            }

        # Run Bayesian optimization
        trials = Trials()
        if verbose:
            print("Starting Bayesian optimization...")
            print("-" * 80)

        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
            verbose=verbose,
            rstate=np.random.default_rng(42),
        )

        # Convert best parameters to correct types
        best_params = space_eval(space, best)
        best_params['n_estimators'] = int(best_params['n_estimators'])
        best_params['max_depth'] = int(best_params['max_depth'])
        best_params['min_child_weight'] = int(best_params['min_child_weight'])
        best_params['random_state'] = 42

        # Get best score from trials
        best_trial = trials.best_trial
        best_cv_score = -best_trial['result']['loss']

        # Train final model with best params and evaluate on test set
        final_model = xgb.XGBRegressor(
            **best_params,
            objective='reg:squarederror',
            tree_method='hist',
            n_jobs=-1,
            verbosity=0,
        )
        final_model.fit(X_train, y_train, verbose=False)
        y_test_pred = final_model.predict(X_test)

        # Convert from log space
        y_test_pred = 10 ** y_test_pred
        y_test_orig = 10 ** y_test
        test_mse = mean_squared_error(y_test_orig, y_test_pred)

        if verbose:
            print("\n" + "=" * 80)
            print("OPTIMIZATION COMPLETE")
            print("=" * 80)
            print(f"Best cross-validation MSE (log): {best_cv_score:.6f}")
            print(f"Test set MSE (original): {test_mse:.4e}")
            print()
            print("Optimal Hyperparameters:")
            for key, value in best_params.items():
                if key != 'random_state':
                    if isinstance(value, float):
                        print(f"  {key:25s}: {value:.6f}")
                    else:
                        print(f"  {key:25s}: {value}")
            print("=" * 80)

        return {
            'best_params': best_params,
            'best_cv_score': best_cv_score,
            'test_mse': test_mse,
            'trials': trials,
        }

    def train(
        self,
        df: pd.DataFrame,
        target_column: str = 'CrossSection',
        test_size: float = 0.2,
        exclude_columns: Optional[list] = None,
        early_stopping_rounds: Optional[int] = 10,
    ) -> Dict[str, float]:
        """
        Train the XGBoost model.

        Args:
            df: Training data (from NucmlDataset.to_tabular())
            target_column: Name of target column
            test_size: Fraction of data for testing
            exclude_columns: Columns to exclude from features
            early_stopping_rounds: Enable early stopping

        Returns:
            Dictionary of training metrics
        """
        # Prepare features and target
        if exclude_columns is None:
            exclude_columns = [target_column, 'Isotope', 'Reaction']

        # Filter out non-numeric columns (includes pandas sparse arrays)
        numeric_columns = df.select_dtypes(include=[np.number, pd.SparseDtype]).columns
        self.feature_columns = [
            col for col in numeric_columns
            if col not in exclude_columns
        ]

        # Handle sparse DataFrames efficiently (avoid memory explosion)
        X_df = df[self.feature_columns]

        # Check if DataFrame contains sparse arrays
        sparse_columns = [col for col in X_df.columns if isinstance(X_df[col].dtype, pd.SparseDtype)]

        if len(sparse_columns) > 0:
            # Convert to scipy sparse matrix (memory efficient)
            import scipy.sparse as sp

            try:
                # Try to convert sparse columns to scipy sparse matrix
                if len(sparse_columns) == len(X_df.columns):
                    # All columns are sparse - use sparse accessor
                    X = sp.csr_matrix(X_df.sparse.to_coo())
                    print(f"  → Using sparse matrix format (memory efficient)")
                else:
                    # Mixed sparse/dense - convert to dense
                    print(f"  → Converting {len(sparse_columns)} sparse columns to dense (mixed DataFrame)")
                    X = X_df.values
            except (AttributeError, ValueError) as e:
                # Fallback to dense if sparse conversion fails
                print(f"  → Sparse conversion failed, using dense format")
                X = X_df.values
        else:
            X = X_df.values

        y = df[target_column].values

        # Log-transform target for better numerical stability
        y_log = np.log10(y + 1e-10)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_log, test_size=test_size, random_state=self.params['random_state']
        )

        # Train model with optional early stopping
        print(f"Training XGBoost ({self.params['n_estimators']} trees, "
              f"max_depth={self.params['max_depth']})...")

        eval_set = [(X_test, y_test)]

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False,
        )

        # Evaluate
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)

        # Convert back from log space
        y_train_pred = 10 ** y_train_pred
        y_test_pred = 10 ** y_test_pred
        y_train_orig = 10 ** y_train
        y_test_orig = 10 ** y_test

        train_mse = mean_squared_error(y_train_orig, y_train_pred)
        test_mse = mean_squared_error(y_test_orig, y_test_pred)
        train_mae = mean_absolute_error(y_train_orig, y_train_pred)
        test_mae = mean_absolute_error(y_test_orig, y_test_pred)

        # Calculate R² score
        train_r2 = 1 - (np.sum((y_train_orig - y_train_pred) ** 2) /
                        np.sum((y_train_orig - y_train_orig.mean()) ** 2))
        test_r2 = 1 - (np.sum((y_test_orig - y_test_pred) ** 2) /
                       np.sum((y_test_orig - y_test_orig.mean()) ** 2))

        self.metrics = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'best_iteration': self.model.best_iteration if hasattr(self.model, 'best_iteration') else None,
        }

        self.is_trained = True

        print(f"✓ Training complete!")
        print(f"  Test MSE: {test_mse:.4e}")
        print(f"  Test MAE: {test_mae:.4e}")
        print(f"  Test R²: {test_r2:.4f}")

        return self.metrics

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            df: Input features (same format as training)

        Returns:
            Predicted cross sections
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        # Handle sparse DataFrames efficiently
        X_df = df[self.feature_columns]
        is_sparse = any(isinstance(dtype, pd.SparseDtype) for dtype in X_df.dtypes)

        if is_sparse:
            # Convert to scipy sparse matrix
            import scipy.sparse as sp
            X = sp.csr_matrix(X_df.sparse.to_coo())
        else:
            X = X_df.values

        y_pred_log = self.model.predict(X)
        y_pred = 10 ** y_pred_log

        return y_pred

    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """
        Get feature importance from the trained model.

        Args:
            importance_type: 'gain', 'weight', or 'cover'
                - gain: Average gain across all splits (recommended)
                - weight: Number of times feature is used
                - cover: Average coverage of splits

        Returns:
            DataFrame with feature importances

        Educational Note:
            Compare this with physics intuition:
            - Energy should be highly important
            - Q-value and threshold should matter for physics mode
            - If Z/A dominate, model is just memorizing isotopes!
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained first")

        # Get importance scores
        importance_dict = self.model.get_booster().get_score(importance_type=importance_type)

        # Map to feature names
        feature_importance = []
        for i, feat_name in enumerate(self.feature_columns):
            # XGBoost uses 'f0', 'f1', ... as feature names internally
            xgb_feat_name = f'f{i}'
            importance = importance_dict.get(xgb_feat_name, 0.0)
            feature_importance.append({'Feature': feat_name, 'Importance': importance})

        df_importance = pd.DataFrame(feature_importance).sort_values('Importance', ascending=False)

        return df_importance

    def predict_resonance_region(
        self,
        Z: int,
        A: int,
        mt_code: int,
        energy_range: Tuple[float, float],
        num_points: int = 1000,
        mode: str = 'naive',
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict cross section in a resonance region.

        Shows that XGBoost is smoother than Decision Tree
        but still not physics-compliant.

        Args:
            Z: Atomic number
            A: Mass number
            mt_code: Reaction type
            energy_range: (E_min, E_max) in eV
            num_points: Number of energy points
            mode: Feature mode ('naive' or 'physics')

        Returns:
            (energies, predictions)
        """
        energies = np.linspace(energy_range[0], energy_range[1], num_points)

        # Build feature matrix
        if mode == 'naive':
            # One-hot encoding
            features = []
            for energy in energies:
                feat_dict = {'Z': Z, 'A': A, 'Energy': energy}
                # Add MT one-hot
                for mt in [2, 16, 18, 102]:
                    feat_dict[f'MT_{mt}'] = 1.0 if mt == mt_code else 0.0
                features.append(feat_dict)
            df = pd.DataFrame(features)

        else:  # physics mode
            features = []
            for energy in energies:
                feat_dict = {
                    'Z': Z,
                    'A': A,
                    'N': A - Z,
                    'Energy': np.log10(energy + 1.0),
                    'Q_Value': 0.0,
                    'Threshold': 0.0,
                    'Delta_Z': 0,
                    'Delta_A': 0,
                    'MT': mt_code / 100.0,
                }
                features.append(feat_dict)
            df = pd.DataFrame(features)

        # Ensure all training features are present
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0.0

        # Predict
        predictions = self.predict(df)

        return energies, predictions

    def save(self, filepath: str) -> None:
        """Save model to disk."""
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")

        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'metrics': self.metrics,
            'params': self.params,
        }
        joblib.dump(model_data, filepath)
        print(f"✓ Model saved to {filepath}")

    def load(self, filepath: str) -> None:
        """Load model from disk."""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.feature_columns = model_data['feature_columns']
        self.metrics = model_data['metrics']
        self.params = model_data['params']
        self.is_trained = True
        print(f"✓ Model loaded from {filepath}")

    def analyze_predictions(
        self,
        df: pd.DataFrame,
        target_column: str = 'CrossSection',
    ) -> Dict[str, Any]:
        """
        Analyze prediction quality across different regimes.

        Args:
            df: Test data with ground truth
            target_column: Name of target column

        Returns:
            Dictionary with analysis results

        Educational Analysis:
            - Low energy (thermal): XGBoost usually good
            - Resonance region: May show artifacts
            - High energy (fast): Extrapolation poor
        """
        predictions = self.predict(df)
        y_true = df[target_column].values

        # Split into energy regimes
        thermal_mask = df['Energy'] < 1.0  # < 1 eV
        resonance_mask = (df['Energy'] >= 1.0) & (df['Energy'] < 1e3)  # 1 eV - 1 keV
        fast_mask = df['Energy'] >= 1e3  # > 1 keV

        analysis = {}

        for name, mask in [('thermal', thermal_mask), ('resonance', resonance_mask), ('fast', fast_mask)]:
            if mask.sum() > 0:
                y_true_regime = y_true[mask]
                y_pred_regime = predictions[mask]

                mse = mean_squared_error(y_true_regime, y_pred_regime)
                mae = mean_absolute_error(y_true_regime, y_pred_regime)
                max_error = np.max(np.abs(y_true_regime - y_pred_regime))

                analysis[name] = {
                    'mse': mse,
                    'mae': mae,
                    'max_error': max_error,
                    'num_points': mask.sum(),
                }

        return analysis
