import warnings
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from statsmodels.stats.outliers_influence import variance_inflation_factor


# Function by Prashant: https://stats.stackexchange.com/questions/155028/how-to-systematically-remove-collinear-variables-pandas-columns-in-python
def calculate_vif(X: pd.DataFrame,  # noqa
                  thresh: float = 10.0,
                  verbose: bool = False) -> pd.DataFrame:
    """
    Calculate the Variance Inflation Factor (VIF) for each independent variable in the dataset and
    remove variables with a VIF above a specified threshold.

    Args:
        X (pd.DataFrame): The input DataFrame containing all independent variables for which VIFs are to be calculated.
                          The DataFrame is expected to have numeric columns only.
        thresh (float, optional): The VIF threshold above which an independent variable will be removed from the dataset.
                                  Defaults to 10.0, which is a common heuristic for identifying significant multicollinearity.
        verbose (bool, optional): Whether to print which columns is getting dropped and which is kept. Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame with variables having a VIF less than the specified threshold.
        The constant column added for VIF calculation is removed from the output.

    Note:
        This function adds a constant column to `X` to calculate the VIF,
        which is a common practice in regression analysis to include an intercept term. The constant column is not included in the returned DataFrame.
    """

    X = X.assign(const=1)  # faster than add_constant from statsmodels
    variables = list(range(X.shape[1]))
    dropped = True
    while dropped:
        dropped = False
        vif = [variance_inflation_factor(X.iloc[:, variables].values, ix)
               for ix in range(X.iloc[:, variables].shape[1])]
        vif = vif[:-1]  # don't let the constant be removed in the loop.
        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            if verbose:
                print(f"dropping '{X.iloc[:, variables].columns[maxloc]}' at index: {maxloc}")
            del variables[maxloc]
            dropped = True
    if verbose:
        print('Remaining variables:')
        print(X.columns[variables[:-1]])

    return X.iloc[:, variables[:-1]]



def select_features_boruta(X: pd.DataFrame,  # noqa
                           y: pd.DataFrame,
                           max_depth: int = 5,
                           n_jobs: int = -1,
                           verbose: int = 0,
                           random_state: int = 42) -> pd.DataFrame:
    """
    Perform feature selection using Boruta on a given DataFrame.

    Parameters:
    - X: pandas DataFrame, feature set
    - y: array-like, target variable
    - max_depth: int, maximum depth of the tree in RandomForestRegressor
    - n_jobs: int, number of jobs to run in parallel for RandomForestRegressor
    - verbose: int, verbosity level for BorutaPy
    - random_state: int, random state for reproducibility

    Returns:
    - pandas DataFrame of the selected features (confirmed and tentative)

    Notes:
        For borutaPy to work ensure you run numpy==1.23.5 as boruta==0.3 has not been updated after np.int() has been deprecated.
    """

    rf = RandomForestRegressor(max_depth=max_depth, n_jobs=n_jobs, random_state=random_state)
    feat_selector = BorutaPy(rf, n_estimators='auto', verbose=verbose, random_state=random_state)

    X_np = X.to_numpy()
    feat_selector.fit(X_np, y)

    # Create a mask for both confirmed and tentative features
    features_mask = feat_selector.support_ | feat_selector.support_weak_

    # Apply the mask to the numpy array to get the filtered features including tentative ones
    X_filtered_np = X_np[:, features_mask]

    # Get the column names for confirmed and tentative features from the original DataFrame
    filtered_feature_names = X.columns[features_mask]
    X_filtered_df = pd.DataFrame(X_filtered_np, columns=filtered_feature_names)

    return X_filtered_df


class VotingSelector:
    def __init__(self,
                 n_features_to_select: int = 25,
                 voting_threshold: float = 0.5,
                 vif_threshold: float = 10) -> None:
        """
        Initialize the VotingSelector with specified feature selection settings.

        Args:
          n_features_to_select (int): The maximum number of features to select for spearman and rfe
          voting_threshold (float): The threshold for a feature to be selected based on voting.
          vif_threshold (float): The threshold for excluding features based on variance inflation factor (VIF).

        Attributes:
          selectors (Dict[str, Callable]): A dictionary mapping selection method names to their corresponding functions.
          votes (Optional[pd.DataFrame]): A DataFrame storing the votes for each feature by each selector. Initialized as None.
        """
        self.n_features_to_select = n_features_to_select
        self.voting_threshold = voting_threshold
        self.vif_threshold = vif_threshold
        self.correlation_method = 'kendall'
        self.selected_features = {}
        self.selectors = {
            "correlation": self._select_correlation,
            "vif": self._select_vif,
            "rfe": self._select_rfe,
            "boruta": self._select_boruta
        }
        self.votes = None

    @staticmethod
    def _remove_constant_columns(X: pd.DataFrame) -> pd.DataFrame: # noqa
        """Remove constant columns from a DataFrame."""
        # A column with zero variance is constant
        return X.loc[:, X.nunique() > 1]


    def _select_correlation(self,
                            X: pd.DataFrame, # noqa
                            y: pd.Series,
                            **kwargs) -> pd.Index:
        """
        Select features based on Spearman correlation with the target.

        Args:
          X (pd.DataFrame): The feature matrix.
          y (pd.Series): The target variable.

        Returns:
          pd.Index: Index of the top features selected based on Spearman correlation.
        """
        # Calculate Spearman correlation between each feature and the target
        corr = X.corrwith(y, method=self.correlation_method).abs()
        sorted_features = corr.sort_values(ascending=False)
        top_features = sorted_features.head(self.n_features_to_select).index
        self.selected_features['correlation'] = top_features
        return top_features


    def _select_vif(self,
                    X: pd.DataFrame, #noqa
                    y: pd.Series,
                    **kwargs) -> List[str]:
        """
        Select features based on the Variance Inflation Factor (VIF).

        Args:
          X (pd.DataFrame): The feature matrix.
          y (pd.Series): The target variable, not used in this method but included for consistency.

        Returns:
          List[str]: List of feature names with VIF below the specified threshold.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            X = X.assign(const=1)
            variables = list(range(X.shape[1]))
            dropped = True
            while dropped and len(variables) > 1:  # Ensure at least one variable remains
                dropped = False
                vif = [variance_inflation_factor(X.iloc[:, variables].values, ix) for ix in range(X.iloc[:, variables].shape[1])]
                maxloc = vif.index(max(vif[:-1]))  # Exclude the constant term from being dropped
                if max(vif[:-1]) > self.vif_threshold:
                    del variables[maxloc]
                    dropped = True
        features_to_keep = X.columns[variables[:-1]].tolist()
        self.selected_features['vif'] = features_to_keep
        return features_to_keep

    def _select_rfe(self,
                    X: pd.DataFrame, # noqa
                    y: pd.Series,
                    **kwargs) -> np.ndarray:
        """
        Select features using Recursive Feature Elimination (RFE).

        Args:
          X (pd.DataFrame): The feature matrix.
          y (pd.Series): The target variable.

        Returns:
          np.ndarray: Array of feature names selected by RFE.
        """
        svr = SVR(kernel="linear")
        rfe = RFE(svr, n_features_to_select=self.n_features_to_select)
        rfe.fit(X, y)
        features_to_keep = rfe.get_feature_names_out()
        self.selected_features['rfe'] = features_to_keep
        return features_to_keep


    def _select_boruta(self, # noqa
                       X: pd.DataFrame, # noqa
                       y: np.ndarray,
                       **kwargs) -> np.ndarray:
        """
        Select features using the Boruta algorithm.

        Args:
          X (pd.DataFrame): The feature matrix.
          y (np.ndarray): The target variable.

        Returns:
          np.ndarray: Array of feature names selected by Boruta.
        """
        rf = RandomForestRegressor(max_depth=None,
                                   n_jobs = -1,
                                   random_state=42)
        feat_selector = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=42)

        X_np = X.to_numpy()
        feat_selector.fit(X_np, y)

        # Create a mask for both confirmed and tentative features
        features_mask = feat_selector.support_ | feat_selector.support_weak_
        features_to_keep = X.columns[features_mask]
        self.selected_features['boruta'] = features_to_keep
        return features_to_keep


    def select(self,
               X: pd.DataFrame, # noqa
               y: pd.Series,
               **kwargs) -> pd.DataFrame:
        """
        Select features based on voting from multiple selection methods.

        Args:
          X (pd.DataFrame): The feature matrix.
          y (pd.Series): The target variable.

        Returns:
          pd.DataFrame: The DataFrame containing only the selected features.
        """
        # Preprocess to remove constant columns
        X_filtered = self._remove_constant_columns(X)

        votes = []
        for selector_name, selector_method in self.selectors.items():
            features_to_keep = selector_method(X_filtered, y, **kwargs)
            vote = pd.Series([int(feature in features_to_keep) for feature in X_filtered.columns], index=X_filtered.columns, name=selector_name)
            votes.append(vote)

        self.votes = pd.DataFrame(votes)
        features_to_keep = list(self.votes.columns[(self.votes.mean() >= self.voting_threshold)])
        return X[features_to_keep]
