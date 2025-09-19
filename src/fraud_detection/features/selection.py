import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif, RFE, RFECV,
    VarianceThreshold, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')


class FraudFeatureSelector:
    """Feature selection methods for fraud detection."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.selected_features = {}
        self.feature_scores = {}
        self.selectors = {}
    
    def remove_low_variance_features(
        self, 
        X: pd.DataFrame, 
        threshold: float = 0.01
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Remove features with low variance."""
        selector = VarianceThreshold(threshold=threshold)
        X_selected = selector.fit_transform(X)
        
        selected_features = X.columns[selector.get_support()].tolist()
        removed_features = X.columns[~selector.get_support()].tolist()
        
        self.selectors['variance'] = selector
        self.selected_features['variance'] = selected_features
        
        print(f"Removed {len(removed_features)} low variance features")
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index), removed_features
    
    def correlation_filter(
        self, 
        X: pd.DataFrame, 
        threshold: float = 0.95
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Remove highly correlated features."""
        corr_matrix = X.corr().abs()
        
        # Create mask for upper triangle
        upper_triangle = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        
        # Find features to remove
        to_remove = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > threshold:
                    # Remove feature with lower variance
                    var_i = X.iloc[:, i].var()
                    var_j = X.iloc[:, j].var()
                    feature_to_remove = corr_matrix.columns[i if var_i < var_j else j]
                    if feature_to_remove not in to_remove:
                        to_remove.append(feature_to_remove)
        
        selected_features = [col for col in X.columns if col not in to_remove]
        self.selected_features['correlation'] = selected_features
        
        print(f"Removed {len(to_remove)} highly correlated features")
        
        return X[selected_features], to_remove
    
    def univariate_selection(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        k: int = 50,
        score_func: str = 'f_classif'
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Select k best features using univariate statistical tests."""
        
        score_functions = {
            'f_classif': f_classif,
            'mutual_info': mutual_info_classif
        }
        
        selector = SelectKBest(
            score_func=score_functions[score_func], 
            k=min(k, X.shape[1])
        )
        
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        # Get feature scores
        feature_scores = dict(zip(
            X.columns[selector.get_support()], 
            selector.scores_[selector.get_support()]
        ))
        
        self.selectors[f'univariate_{score_func}'] = selector
        self.selected_features[f'univariate_{score_func}'] = selected_features
        self.feature_scores[f'univariate_{score_func}'] = feature_scores
        
        print(f"Selected {len(selected_features)} features using {score_func}")
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index), feature_scores
    
    def recursive_feature_elimination(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        n_features: int = 30,
        estimator_type: str = 'logistic'
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Recursive feature elimination with cross-validation."""
        
        estimators = {
            'logistic': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=50, random_state=42)
        }
        
        estimator = estimators[estimator_type]
        
        selector = RFECV(
            estimator=estimator,
            step=1,
            cv=3,
            scoring='roc_auc',
            n_jobs=-1,
            min_features_to_select=min(n_features, X.shape[1])
        )
        
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        self.selectors[f'rfe_{estimator_type}'] = selector
        self.selected_features[f'rfe_{estimator_type}'] = selected_features
        
        print(f"RFE with {estimator_type} selected {len(selected_features)} features")
        print(f"Optimal number of features: {selector.n_features_}")
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index), selected_features
    
    def model_based_selection(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        estimator_type: str = 'random_forest',
        threshold: str = 'median'
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Select features based on model feature importance."""
        
        estimators = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'logistic': LogisticRegression(random_state=42, max_iter=1000, C=0.1)
        }
        
        estimator = estimators[estimator_type]
        estimator.fit(X, y)
        
        selector = SelectFromModel(estimator, threshold=threshold)
        X_selected = selector.fit_transform(X, y)
        
        selected_features = X.columns[selector.get_support()].tolist()
        
        # Get feature importance scores
        if hasattr(estimator, 'feature_importances_'):
            importance_scores = dict(zip(
                X.columns[selector.get_support()],
                estimator.feature_importances_[selector.get_support()]
            ))
        elif hasattr(estimator, 'coef_'):
            importance_scores = dict(zip(
                X.columns[selector.get_support()],
                np.abs(estimator.coef_[0])[selector.get_support()]
            ))
        else:
            importance_scores = {}
        
        self.selectors[f'model_{estimator_type}'] = selector
        self.selected_features[f'model_{estimator_type}'] = selected_features
        self.feature_scores[f'model_{estimator_type}'] = importance_scores
        
        print(f"Model-based selection with {estimator_type} selected {len(selected_features)} features")
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index), importance_scores
    
    def stability_selection(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        n_bootstrap: int = 100,
        threshold: float = 0.6
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Stability selection using bootstrap sampling."""
        
        n_samples, n_features = X.shape
        feature_selection_freq = np.zeros(n_features)
        
        for i in range(n_bootstrap):
            # Bootstrap sampling
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X.iloc[indices]
            y_boot = y.iloc[indices]
            
            # Feature selection on bootstrap sample
            rf = RandomForestClassifier(n_estimators=50, random_state=i)
            rf.fit(X_boot, y_boot)
            
            # Select top 50% features
            importances = rf.feature_importances_
            top_features = np.argsort(importances)[-int(n_features * 0.5):]
            
            feature_selection_freq[top_features] += 1
        
        # Normalize frequencies
        selection_frequencies = feature_selection_freq / n_bootstrap
        
        # Select stable features
        stable_features_mask = selection_frequencies >= threshold
        selected_features = X.columns[stable_features_mask].tolist()
        
        stability_scores = dict(zip(
            selected_features,
            selection_frequencies[stable_features_mask]
        ))
        
        self.selected_features['stability'] = selected_features
        self.feature_scores['stability'] = stability_scores
        
        print(f"Stability selection selected {len(selected_features)} features")
        
        return X[selected_features], stability_scores
    
    def ensemble_selection(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        methods: List[str] = ['univariate_f_classif', 'rfe_random_forest', 'model_random_forest'],
        voting_threshold: float = 0.5
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """Ensemble feature selection combining multiple methods."""
        
        all_selected_features = []
        method_results = {}
        
        for method in methods:
            if method == 'univariate_f_classif':
                _, _ = self.univariate_selection(X, y, k=50, score_func='f_classif')
                method_results[method] = self.selected_features['univariate_f_classif']
            elif method == 'rfe_random_forest':
                _, _ = self.recursive_feature_elimination(X, y, n_features=30, estimator_type='random_forest')
                method_results[method] = self.selected_features['rfe_random_forest']
            elif method == 'model_random_forest':
                _, _ = self.model_based_selection(X, y, estimator_type='random_forest')
                method_results[method] = self.selected_features['model_random_forest']
        
        # Count votes for each feature
        feature_votes = {}
        for method, features in method_results.items():
            for feature in features:
                feature_votes[feature] = feature_votes.get(feature, 0) + 1
        
        # Select features with enough votes
        min_votes = int(len(methods) * voting_threshold)
        ensemble_features = [
            feature for feature, votes in feature_votes.items() 
            if votes >= min_votes
        ]
        
        self.selected_features['ensemble'] = ensemble_features
        self.feature_scores['ensemble'] = feature_votes
        
        print(f"Ensemble selection selected {len(ensemble_features)} features")
        print(f"Voting threshold: {min_votes}/{len(methods)}")
        
        return X[ensemble_features], feature_votes
    
    def select_features(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        method: str = 'ensemble',
        **kwargs
    ) -> pd.DataFrame:
        """
        Main feature selection method.
        
        Args:
            X: Feature dataframe
            y: Target series
            method: Selection method ('ensemble', 'univariate', 'rfe', 'model', 'stability')
            **kwargs: Additional arguments for specific methods
        
        Returns:
            Selected features dataframe
        """
        
        # Remove low variance features first
        X_filtered, _ = self.remove_low_variance_features(X, threshold=0.01)
        
        # Remove highly correlated features
        X_filtered, _ = self.correlation_filter(X_filtered, threshold=0.95)
        
        if method == 'ensemble':
            X_selected, _ = self.ensemble_selection(X_filtered, y, **kwargs)
        elif method == 'univariate':
            X_selected, _ = self.univariate_selection(X_filtered, y, **kwargs)
        elif method == 'rfe':
            X_selected, _ = self.recursive_feature_elimination(X_filtered, y, **kwargs)
        elif method == 'model':
            X_selected, _ = self.model_based_selection(X_filtered, y, **kwargs)
        elif method == 'stability':
            X_selected, _ = self.stability_selection(X_filtered, y, **kwargs)
        else:
            raise ValueError(f"Unknown selection method: {method}")
        
        return X_selected
    
    def get_feature_ranking(self, method: str = 'ensemble') -> pd.DataFrame:
        """Get feature ranking based on selection method."""
        
        if method not in self.feature_scores:
            raise ValueError(f"No scores available for method: {method}")
        
        scores = self.feature_scores[method]
        ranking_df = pd.DataFrame([
            {'feature': feature, 'score': score, 'rank': rank + 1}
            for rank, (feature, score) in enumerate(
                sorted(scores.items(), key=lambda x: x[1], reverse=True)
            )
        ])
        
        return ranking_df
    
    def save_selection_results(self, filepath: str):
        """Save feature selection results."""
        results = {
            'selected_features': self.selected_features,
            'feature_scores': self.feature_scores
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Selection results saved to {filepath}")


def quick_feature_selection(X: pd.DataFrame, y: pd.Series, n_features: int = 30) -> pd.DataFrame:
    """Quick feature selection for rapid prototyping."""
    selector = FraudFeatureSelector()
    
    # Use univariate selection for speed
    X_selected, _ = selector.univariate_selection(X, y, k=n_features)
    
    return X_selected


def comprehensive_feature_selection(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Comprehensive feature selection using ensemble approach."""
    selector = FraudFeatureSelector()
    
    X_selected = selector.select_features(X, y, method='ensemble')
    
    return X_selected